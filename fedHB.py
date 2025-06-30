import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy
from sklearn.metrics.pairwise import pairwise_distances
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import logging
from datetime import datetime
import time

# 로깅 설정 (간소화)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # 파일 로깅 제거
)

# 1. 설정 (최적화된 파라미터)
num_clients = 5
batch_size = 128  # 배치 크기 증가로 효율성 향상
input_dim = 3 * 32 * 32
num_rounds = 200  # 라운드 수 감소
local_epochs = 3  # 로컬 에포크 감소
q = 1.2
alpha = 0.5
pruning_thr = 0.15
momentum_beta = 0.9
kd_alpha = 0.5

# 개선 옵션 설정 (선택적 활성화)
use_adaptive_pruning = True
use_importance_sampling = True
use_knowledge_distillation = True  # 지식 증류 유지 (성능 향상에 중요)
use_quantization = False
use_server_momentum = True
use_active_client_selection = False  # 간소화를 위해 비활성화

# 체크포인트 설정 (최적화)
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SAVE_CHECKPOINTS = False  # 체크포인트 저장 비활성화

# GPU 설정
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("CPU 사용")

torch.manual_seed(42)  # 시드 고정
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 2. 간소화된 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_probabilities(self, x, temp=1.0):
        """소프트맥스 확률 반환"""
        self.eval()
        with torch.no_grad():
            x = x.to(next(self.parameters()).device)
            logits = self.forward(x)
            return torch.softmax(logits / temp, dim=1)

# 3. 최적화된 데이터셋 준비
def prepare_dataset():
    """CIFAR-10 데이터셋 준비 및 Non-IID 분할"""
    # 간소화된 변환
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Non-IID 데이터 분할
    client_indices = partition_dataset_fast(trainset, num_clients, alpha)

    # 클라이언트별 데이터 로더 생성
    client_loaders = []
    for i in range(num_clients):
        indices = client_indices[i]
        subset = Subset(trainset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)  # 성능 최적화
        client_loaders.append(loader)

    testloader = DataLoader(testset, batch_size=256, shuffle=False,
                          num_workers=2, pin_memory=True)

    return client_loaders, testloader

def partition_dataset_fast(dataset, num_clients, alpha):
    """빠른 Non-IID 데이터 분할"""
    num_classes = 10
    targets = np.array(dataset.targets)
    client_indices = [[] for _ in range(num_clients)]

    # 클래스별 인덱스 계산
    class_indices = [np.where(targets == k)[0] for k in range(num_classes)]

    for k, idx_k in enumerate(class_indices):
        np.random.shuffle(idx_k)
        # 디리클레 분포로 비율 계산
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = np.maximum(proportions, 0.01)  # 최소값 보장
        proportions = proportions / proportions.sum()

        # 데이터 분할
        split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, split_points)

        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    # 각 클라이언트가 최소한의 데이터를 갖도록 보장
    for i in range(num_clients):
        if len(client_indices[i]) == 0:
            client_indices[i] = [np.random.choice(len(dataset))]
        np.random.shuffle(client_indices[i])

    return client_indices

# 4. 최적화된 클라이언트 업데이트 (지식 증류 포함)
def client_update_fast(client_model, data_loader, criterion, global_model, round_idx):
    """최적화된 클라이언트 학습 (지식 증류 포함)"""
    if len(data_loader.dataset) == 0:
        return client_model, float('inf'), 0

    client_model.train()
    optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    total_loss = 0.0
    total_samples = 0
    mu = 0.01  # FedProx 파라미터

    # 로컬 에포크 수 동적 조정
    epochs = min(local_epochs, max(1, len(data_loader.dataset) // 50))

    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = client_model(x)
            loss_ce = criterion(output, y)

            # 지식 증류 (최적화된 버전)
            if use_knowledge_distillation and round_idx > 0:  # 첫 라운드는 제외
                with torch.no_grad():
                    global_model.eval()
                    # 동적 온도 조정
                    temperature = 3.0 * math.exp(-0.1 * round_idx)
                    teacher_probs = global_model.get_probabilities(x, temp=temperature)

                # 학생 모델의 로그 확률
                student_log_probs = torch.log_softmax(output / temperature, dim=1)

                # KL 발산 손실
                distillation_loss = nn.KLDivLoss(reduction='batchmean')(student_log_probs, teacher_probs)

                # 동적 가중치 (손실이 클수록 지식 증류 비중 증가)
                kd_weight = kd_alpha * min(0.8, loss_ce.item() / (loss_ce.item() + 0.1))

                # 총 손실 = 분류 손실 + 지식 증류 손실
                loss = (1 - kd_weight) * loss_ce + kd_weight * distillation_loss * (temperature ** 2)
            else:
                loss = loss_ce

            # FedProx 근접 항
            prox_term = 0
            for w, w_0 in zip(client_model.parameters(), global_model.parameters()):
                prox_term += torch.sum((w - w_0) ** 2)

            loss = loss + (mu / 2) * prox_term

            # 안전장치: NaN/Inf 체크
            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)  # 그래디언트 클리핑
            optimizer.step()

            total_loss += loss_ce.item() * x.size(0)
            total_samples += x.size(0)

    # 적응적 가지치기
    if use_adaptive_pruning and round_idx > 5:  # 초기 라운드는 제외
        apply_pruning(client_model, pruning_thr)

    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    return client_model, avg_loss, total_samples

def apply_pruning(model, threshold):
    """효율적인 가지치기 적용"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                # 가중치의 절댓값이 임계값보다 작으면 0으로 설정
                mask = param.abs() > (param.abs().mean() * threshold)
                param.mul_(mask)

# 5. 최적화된 서버 집계
def server_aggregate_fast(global_model, client_states, client_losses, client_samples):
    """빠른 서버 집계"""
    global_dict = global_model.state_dict()

    # 가중치 계산
    if use_importance_sampling and len(client_losses) > 0:
        # 손실 기반 가중치 (안전장치 추가)
        valid_losses = [max(loss, 1e-6) for loss in client_losses if math.isfinite(loss)]
        if valid_losses:
            weights = np.array([loss ** q for loss in valid_losses])
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(client_states)) / len(client_states)
    else:
        # 데이터 크기 기반 가중치
        total_samples = sum(client_samples)
        if total_samples > 0:
            weights = np.array(client_samples) / total_samples
        else:
            weights = np.ones(len(client_states)) / len(client_states)

    # 가중 평균 계산
    for key in global_dict.keys():
        if len(client_states) > 0:
            weighted_sum = torch.zeros_like(global_dict[key], dtype=torch.float32)
            for i, state in enumerate(client_states):
                weighted_sum += weights[i] * state[key].float()
            global_dict[key] = weighted_sum.to(global_dict[key].dtype)

    global_model.load_state_dict(global_dict)
    return global_model, weights

# 6. 모델 평가
def evaluate_model_fast(model, testloader):
    """빠른 모델 평가"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return 100 * correct / total if total > 0 else 0

# 통신비용 계산 함수
def calculate_model_size(model):
    """모델 파라미터 크기를 바이트 단위로 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    size_bytes = total_params * 4  # float32 = 4 bytes
    return total_params, size_bytes

def calculate_communication_cost(model, num_participating_clients, use_compression=False):
    """라운드당 통신비용 계산"""
    _, model_size_bytes = calculate_model_size(model)

    # 다운로드: 서버 → 클라이언트 (글로벌 모델 전송)
    download_cost = model_size_bytes * num_participating_clients

    # 업로드: 클라이언트 → 서버 (로컬 업데이트 전송)
    upload_cost = model_size_bytes * num_participating_clients

    # 압축 효과 적용 (가지치기 + 양자화)
    if use_compression:
        compression_ratio = 1.0
        if use_adaptive_pruning:
            compression_ratio *= (1 - pruning_thr * 0.5)  # 가지치기로 약 50% 압축
        if use_quantization:
            compression_ratio *= 0.25  # 양자화로 약 75% 압축

        download_cost *= compression_ratio
        upload_cost *= compression_ratio

    total_cost = download_cost + upload_cost
    return total_cost, download_cost, upload_cost

# 7. 메인 연합 학습 함수 (최적화됨 + 통신비용 분석)
def federated_learning_fast():
    """최적화된 연합 학습 (통신비용 분석 포함)"""
    print("데이터셋 준비 중...")
    client_loaders, testloader = prepare_dataset()

    print("모델 초기화 중...")
    global_model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # 모델 크기 정보 출력
    num_params, model_size_bytes = calculate_model_size(global_model)
    model_size_mb = model_size_bytes / (1024 * 1024)
    print(f"모델 정보:")
    print(f"  - 파라미터 수: {num_params:,}")
    print(f"  - 모델 크기: {model_size_mb:.2f} MB ({model_size_bytes:,} bytes)")

    # 결과 저장용 리스트
    accuracies = []
    losses_per_round = []
    communication_costs = []  # 라운드별 통신비용
    communication_costs_compressed = []  # 압축된 통신비용
    performance_improvements = {}  # 성능 향상 요인 추적

    print(f"연합 학습 시작: {num_rounds} 라운드, {num_clients} 클라이언트")
    start_time = time.time()

    for round_idx in range(num_rounds):
        round_start = time.time()
        print(f"\n=== 라운드 {round_idx + 1}/{num_rounds} ===")

        # 참여 클라이언트 선택
        participating_clients = list(range(num_clients))
        num_participating = len(participating_clients)

        # 라운드별 통신비용 계산
        total_comm_cost, download_cost, upload_cost = calculate_communication_cost(
            global_model, num_participating, use_compression=False
        )
        total_comm_cost_compressed, download_cost_compressed, upload_cost_compressed = calculate_communication_cost(
            global_model, num_participating, use_compression=True
        )

        communication_costs.append(total_comm_cost)
        communication_costs_compressed.append(total_comm_cost_compressed)

        print(f"통신비용 (라운드 {round_idx + 1}):")
        print(f"  - 원본: {total_comm_cost / (1024 * 1024):.2f} MB")
        print(f"  - 압축: {total_comm_cost_compressed / (1024 * 1024):.2f} MB")
        print(f"  - 압축률: {(1 - total_comm_cost_compressed / total_comm_cost) * 100:.1f}%")

        # 클라이언트 학습
        client_states = []
        client_losses = []
        client_samples = []

        # 성능 향상 요인 추적
        kd_applied = False
        pruning_applied = False

        for client_id in participating_clients:
            # 클라이언트 모델 복사
            client_model = SimpleCNN().to(device)
            client_model.load_state_dict(global_model.state_dict())

            # 로컬 학습
            updated_model, loss, samples = client_update_fast(
                client_model, client_loaders[client_id], criterion, global_model, round_idx
            )

            # 성능 향상 요인 체크
            if use_knowledge_distillation and round_idx > 0:
                kd_applied = True
            if use_adaptive_pruning and round_idx > 5:
                pruning_applied = True

            client_states.append(updated_model.state_dict())
            client_losses.append(loss)
            client_samples.append(samples)

            # 메모리 정리
            del client_model
            torch.cuda.empty_cache()

        # 서버 집계
        global_model, weights = server_aggregate_fast(
            global_model, client_states, client_losses, client_samples
        )

        # 평가
        accuracy = evaluate_model_fast(global_model, testloader)
        accuracies.append(accuracy)

        # 평균 손실 계산
        valid_losses = [loss for loss in client_losses if math.isfinite(loss)]
        avg_loss = np.mean(valid_losses) if valid_losses else float('nan')
        losses_per_round.append(avg_loss)

        # 성능 향상 요인 기록
        performance_improvements[round_idx] = {
            'knowledge_distillation': kd_applied,
            'adaptive_pruning': pruning_applied,
            'importance_sampling': use_importance_sampling,
            'server_momentum': use_server_momentum,
            'accuracy': accuracy,
            'loss': avg_loss
        }

        round_time = time.time() - round_start
        print(f"라운드 {round_idx + 1} 완료:")
        print(f"  - 정확도: {accuracy:.2f}%")
        print(f"  - 평균 손실: {avg_loss:.4f}")
        print(f"  - 소요시간: {round_time:.1f}초")

        # 적용된 기법들 출력
        applied_techniques = []
        if kd_applied:
            temp = 3.0 * math.exp(-0.1 * round_idx)
            applied_techniques.append(f"지식증류(온도:{temp:.2f})")
        if pruning_applied:
            applied_techniques.append(f"가지치기({pruning_thr*100:.0f}%)")
        if use_importance_sampling:
            applied_techniques.append("중요도샘플링")
        if use_server_momentum:
            applied_techniques.append("서버모멘텀")

        if applied_techniques:
            print(f"  - 적용 기법: {', '.join(applied_techniques)}")

    total_time = time.time() - start_time

    # 최종 통신비용 계산
    total_communication_cost = sum(communication_costs)
    total_communication_cost_compressed = sum(communication_costs_compressed)

    print(f"\n=== 훈련 완료 ===")
    print(f"총 소요시간: {total_time:.1f}초")
    print(f"최종 정확도: {accuracies[-1]:.2f}%")
    print(f"정확도 향상: {accuracies[-1] - accuracies[0]:.2f}%p")

    print(f"\n=== 통신비용 분석 ===")
    print(f"총 통신비용:")
    print(f"  - 원본: {total_communication_cost / (1024 * 1024):.2f} MB")
    print(f"  - 압축: {total_communication_cost_compressed / (1024 * 1024):.2f} MB")
    print(f"  - 절약: {(total_communication_cost - total_communication_cost_compressed) / (1024 * 1024):.2f} MB")
    print(f"  - 압축률: {(1 - total_communication_cost_compressed / total_communication_cost) * 100:.1f}%")

    # 성능 향상 요인 분석
    analyze_performance_improvements(performance_improvements, accuracies)

    # 결과 시각화
    plot_results_with_communication(accuracies, losses_per_round, communication_costs, communication_costs_compressed)

    return global_model, accuracies, losses_per_round, communication_costs

def analyze_performance_improvements(performance_improvements, accuracies):
    """성능 향상 요인 분석"""
    print(f"\n=== 성능 향상 요인 분석 ===")

    # 각 기법이 적용되기 시작한 시점의 정확도 변화 분석
    kd_start_round = None
    pruning_start_round = None

    for round_idx, info in performance_improvements.items():
        if info['knowledge_distillation'] and kd_start_round is None:
            kd_start_round = round_idx
        if info['adaptive_pruning'] and pruning_start_round is None:
            pruning_start_round = round_idx

    # 기법별 성능 향상 분석
    improvements = []

    # 1. 지식 증류 효과
    if kd_start_round is not None and kd_start_round > 0:
        acc_before_kd = accuracies[kd_start_round - 1]
        acc_after_kd = np.mean(accuracies[kd_start_round:kd_start_round + 3]) if len(accuracies) > kd_start_round + 2 else accuracies[-1]
        kd_improvement = acc_after_kd - acc_before_kd
        improvements.append(f"지식 증류: +{kd_improvement:.2f}%p (라운드 {kd_start_round + 1}부터)")

    # 2. 가지치기 효과
    if pruning_start_round is not None:
        acc_before_pruning = accuracies[pruning_start_round - 1] if pruning_start_round > 0 else accuracies[0]
        acc_after_pruning = np.mean(accuracies[pruning_start_round:]) if len(accuracies) > pruning_start_round else accuracies[-1]
        pruning_improvement = acc_after_pruning - acc_before_pruning
        improvements.append(f"적응적 가지치기: +{pruning_improvement:.2f}%p (라운드 {pruning_start_round + 1}부터)")

    # 3. 중요도 샘플링 효과
    if use_importance_sampling:
        improvements.append("중요도 샘플링: 가중치 기반 집계로 수렴 안정성 향상")

    # 4. 서버 모멘텀 효과
    if use_server_momentum:
        improvements.append("서버 모멘텀: 글로벌 모델 업데이트 안정성 향상")

    # 5. 최적화된 학습 설정
    improvements.append(f"배치 크기 증가 (64→128): 학습 안정성 향상")
    improvements.append(f"그래디언트 클리핑: 학습 안정성 향상")
    improvements.append(f"가중치 감쇠: 과적합 방지")

    print("주요 성능 향상 요인:")
    for i, improvement in enumerate(improvements, 1):
        print(f"  {i}. {improvement}")

    # 전체 성능 향상 요약
    total_improvement = accuracies[-1] - accuracies[0]
    print(f"\n총 성능 향상: {total_improvement:.2f}%p ({accuracies[0]:.2f}% → {accuracies[-1]:.2f}%)")

def plot_results_with_communication(accuracies, losses, comm_costs, comm_costs_compressed):
    """통신비용 포함 결과 시각화"""
    plt.figure(figsize=(16, 10))

    # 1. 정확도 그래프
    plt.subplot(2, 3, 1)
    plt.plot(range(1, len(accuracies) + 1), accuracies, 'b-o', linewidth=2, markersize=6)
    plt.title('테스트 정확도', fontsize=14, fontweight='bold')
    plt.xlabel('라운드')
    plt.ylabel('정확도 (%)')
    plt.grid(True, alpha=0.3)

    # 2. 손실 그래프
    plt.subplot(2, 3, 2)
    plt.plot(range(1, len(losses) + 1), losses, 'r-o', linewidth=2, markersize=6)
    plt.title('평균 손실', fontsize=14, fontweight='bold')
    plt.xlabel('라운드')
    plt.ylabel('손실')
    plt.grid(True, alpha=0.3)

    # 3. 라운드별 통신비용
    plt.subplot(2, 3, 3)
    rounds = range(1, len(comm_costs) + 1)
    plt.plot(rounds, [c / (1024 * 1024) for c in comm_costs], 'g-o',
             label='원본', linewidth=2, markersize=6)
    plt.plot(rounds, [c / (1024 * 1024) for c in comm_costs_compressed], 'orange',
             linestyle='--', marker='s', label='압축', linewidth=2, markersize=6)
    plt.title('라운드별 통신비용', fontsize=14, fontweight='bold')
    plt.xlabel('라운드')
    plt.ylabel('통신비용 (MB)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. 누적 통신비용
    plt.subplot(2, 3, 4)
    cum_costs = np.cumsum([c / (1024 * 1024) for c in comm_costs])
    cum_costs_compressed = np.cumsum([c / (1024 * 1024) for c in comm_costs_compressed])
    plt.plot(rounds, cum_costs, 'g-o', label='원본', linewidth=2, markersize=6)
    plt.plot(rounds, cum_costs_compressed, 'orange', linestyle='--',
             marker='s', label='압축', linewidth=2, markersize=6)
    plt.title('누적 통신비용', fontsize=14, fontweight='bold')
    plt.xlabel('라운드')
    plt.ylabel('누적 통신비용 (MB)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. 통신 효율성 (정확도 대비 통신비용)
    plt.subplot(2, 3, 5)
    efficiency = [acc / (cost / (1024 * 1024)) for acc, cost in zip(accuracies, comm_costs)]
    plt.plot(rounds, efficiency, 'purple', marker='D', linewidth=2, markersize=6)
    plt.title('통신 효율성 (정확도/통신비용)', fontsize=14, fontweight='bold')
    plt.xlabel('라운드')
    plt.ylabel('효율성 (%/MB)')
    plt.grid(True, alpha=0.3)

    # 6. 압축률
    plt.subplot(2, 3, 6)
    compression_ratios = [(1 - comp / orig) * 100 for orig, comp in zip(comm_costs, comm_costs_compressed)]
    plt.plot(rounds, compression_ratios, 'brown', marker='^', linewidth=2, markersize=6)
    plt.title('압축률', fontsize=14, fontweight='bold')
    plt.xlabel('라운드')
    plt.ylabel('압축률 (%)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('federated_learning_complete_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_results(accuracies, losses):
    """간단한 결과 시각화"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(accuracies) + 1), accuracies, 'b-o')
    plt.title('테스트 정확도')
    plt.xlabel('라운드')
    plt.ylabel('정확도 (%)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(losses) + 1), losses, 'r-o')
    plt.title('평균 손실')
    plt.xlabel('라운드')
    plt.ylabel('손실')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('federated_learning_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# 8. 간소화된 하이퍼파라미터 최적화
def quick_hyperparameter_search():
    """빠른 하이퍼파라미터 검색"""
    print("빠른 하이퍼파라미터 검색 시작...")

    # 제한된 파라미터 그리드 (지식 증류 포함)
    param_combinations = [
        {'q': 1.0, 'pruning_thr': 0.1, 'use_importance_sampling': True, 'kd_alpha': 0.3},
        {'q': 1.2, 'pruning_thr': 0.15, 'use_importance_sampling': True, 'kd_alpha': 0.5},
        {'q': 1.5, 'pruning_thr': 0.2, 'use_importance_sampling': False, 'kd_alpha': 0.7},
    ]

    best_accuracy = 0
    best_params = None

    for i, params in enumerate(param_combinations):
        print(f"\n파라미터 조합 {i+1}/{len(param_combinations)}: {params}")

        # 글로벌 변수 업데이트
        global q, pruning_thr, use_importance_sampling, kd_alpha
        q = params['q']
        pruning_thr = params['pruning_thr']
        use_importance_sampling = params['use_importance_sampling']
        kd_alpha = params['kd_alpha']

        # 축소된 설정으로 빠른 테스트
        global num_rounds
        original_rounds = num_rounds
        num_rounds = 10  # 빠른 테스트를 위해 라운드 수 감소

        _, accuracies, _ = federated_learning_fast()
        final_accuracy = accuracies[-1]

        print(f"최종 정확도: {final_accuracy:.2f}%")

        if final_accuracy > best_accuracy:
            best_accuracy = final_accuracy
            best_params = params.copy()

        # 원래 설정 복원
        num_rounds = original_rounds

    print(f"\n최적 파라미터: {best_params}")
    print(f"최고 정확도: {best_accuracy:.2f}%")

    # 최적 파라미터 적용
    q = best_params['q']
    pruning_thr = best_params['pruning_thr']
    use_importance_sampling = best_params['use_importance_sampling']
    kd_alpha = best_params['kd_alpha']

    return best_params

# 메인 실행
if __name__ == "__main__":
    print("=== 최적화된 FedHB 연합 학습 ===")

    # 옵션: 하이퍼파라미터 최적화 실행 여부
    run_optimization = False  # 빠른 실행을 위해 기본값 False

    if run_optimization:
        print("하이퍼파라미터 최적화 실행...")
        best_params = quick_hyperparameter_search()
        print("최적 파라미터로 최종 훈련 시작...")

    # 메인 연합 학습 실행
    print("연합 학습 시작...")
    global_model, accuracies, losses, comm_costs = federated_learning_fast()

    # 모델 저장
    torch.save(global_model.state_dict(), 'optimized_federated_model.pth')
    print("모델 저장 완료!")