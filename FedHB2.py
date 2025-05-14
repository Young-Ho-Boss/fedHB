import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import concurrent.futures
import matplotlib.pyplot as plt
from collections import OrderedDict
import copy
from sklearn.metrics.pairwise import pairwise_distances
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'federated_learning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# 체크포인트 디렉토리 생성
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 1. 설정
num_clients = 5
batch_size = 64
input_dim = 3 * 32 * 32
num_rounds = 50
local_epochs = 5
q = 1.2  # q 파라미터: 손실 기반 가중치 조정 (q=1: FedAvg, q>1: 손실이 큰 클라이언트 중요도 증가)
alpha = 0.5  # alpha 파라미터: 데이터 분포의 불균형성 (작을수록 극단적 편향)
pruning_thr = 0.15  # 가지치기 임계값
momentum_beta = 0.9  # 서버 모멘텀 계수
kd_alpha = 0.5  # 지식 증류 가중치 계수

# 개선 옵션 설정
use_adaptive_pruning = True  # 적응적 가지치기 사용
use_importance_sampling = True  # 중요도 샘플링 사용
use_knowledge_distillation = True  # 지식 증류 사용
use_quantization = True  # 양자화 사용
use_server_momentum = True  # 서버 모멘텀 사용
use_active_client_selection = True  # 활성 클라이언트 선택 사용
use_async_parallel = False  # 비동기 병렬 처리 비활성화

# 양자화 설정
quantization_bits_conv = 8  # 컨볼루션 레이어 양자화 비트
quantization_bits_fc = 16   # 완전연결 레이어 양자화 비트

# GPU 설정
if not torch.cuda.is_available():
    raise RuntimeError("GPU가 필요합니다. CUDA가 설치되어 있는지 확인해주세요.")
device = torch.device('cuda')
torch.cuda.empty_cache()  # GPU 메모리 초기화
print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
print(f"사용 가능한 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

torch.manual_seed(0)
torch.cuda.manual_seed(0)  # GPU 시드 설정

# 수정 1: Enhancer 레이어 정의 (Classifier 계층)
class EnhancedClassifier(nn.Module):
    def __init__(self, in_features=4096, out_features=10, rank=32):
        super().__init__()
        # 저랭크 파라미터화 (Low-rank Factorization)
        self.A = nn.Parameter(torch.randn(in_features, rank))
        self.B = nn.Parameter(torch.randn(rank, out_features))
        
    def forward(self, x):
        return x @ (self.A @ self.B)  # W = A*B (Low-rank)
    
# 2. 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = EnhancedClassifier()  # in_features=4096, out_features=10, rank=32
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.classifier(x)

# 수정 2: 구조화된 업데이트 적용 (Algorithm 1)
def structured_update(params):
    for name, param in params.items():
        if 'classifier' in name:  # Enhancer 계층만 처리
            # Random Masking (검색 결과[2]의 전략)
            mask = torch.rand_like(param) > 0.8  # 20% 유지
            param.data *= mask.float()
            # 양자화 (검색 결과[5] QSGD 기반)
            param.data = quantize(param.data, bits=4)

# 수정 3: 클라이언트 학습 시 Enhancer만 업데이트
def client_update(client_model, data_loader, optimizer, criterion, global_model, round_idx, total_rounds, pruning_threshold, kd_alpha):
    """클라이언트 로컬 학습 함수"""
    try:
        client_model.train()
        total_loss = 0.0
        total_samples = 0

        # FedProx 파라미터
        mu = 0.01  # 프록시멀 항 계수

        for epoch in range(local_epochs):
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                # 정방향 계산
                output = client_model(x)
                loss_ce = criterion(output, y)

                # 지식 증류 (동적 가중치)
                if use_knowledge_distillation:
                    with torch.no_grad():
                        global_model.eval()
                        teacher_probs = global_model.get_probabilities(x, temp=2.0)

                    student_log_probs = torch.log_softmax(output/2.0, dim=1)
                    distillation_loss = nn.KLDivLoss(reduction='batchmean')(student_log_probs, teacher_probs)

                    # 코사인 스케줄링을 통한 동적 가중치 조절
                    kd_weight = kd_alpha * (1 + math.cos(math.pi * round_idx / total_rounds))
                    loss = (1 - kd_weight) * loss_ce + kd_weight * distillation_loss
                else:
                    loss = loss_ce

                # FedProx 항 추가
                prox_term = 0
                for w, w_0 in zip(client_model.parameters(), global_model.parameters()):
                    prox_term += torch.sum((w - w_0) ** 2)

                # 최종 손실
                loss += (mu / 2) * prox_term

                # 역전파 및 옵티마이저 스텝
                loss.backward()
                optimizer.step()

                total_loss += loss_ce.item() * x.size(0)
                total_samples += x.size(0)

        # 가지치기 (pruning)
        if use_adaptive_pruning:
            with torch.no_grad():
                # 모든 가중치의 절대값 수집
                all_weights = torch.cat([p.view(-1).abs() for p in client_model.parameters() if p.requires_grad])

                # 임계값 계산
                threshold = torch.quantile(all_weights, pruning_threshold)

                # 가지치기 적용
                for name, param in client_model.named_parameters():
                    if 'weight' in name:
                        mask = param.abs() > threshold
                        param.mul_(mask)

        # 양자화 적용
        if use_quantization:
            with torch.no_grad():
                for name, param in client_model.named_parameters():
                    if 'weight' in name:
                        param.copy_(mixed_precision_quantize(param, name))

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        return client_model, avg_loss, total_samples

    finally:
        # 메모리 해제
        torch.cuda.empty_cache()

# 수정 4: 서버 집계 최적화 (Algorithm 2)
def server_aggregate(global_model, client_states, client_losses, client_samples, server_velocity=None, momentum_beta=0.9):
    """클라이언트 모델 집계 및 서버 모델 업데이트"""
    global_dict = global_model.state_dict()

    if use_importance_sampling:
        # 손실 기반 가중치 계산 (높은 손실 = 더 많은 가중치)
        weights = np.array([loss ** q for loss in client_losses])
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
        else:
            # 가중치를 계산할 수 없는 경우 샘플 크기 비례로 가중치 설정
            weights = np.array(client_samples) / np.sum(client_samples) if np.sum(client_samples) > 0 else np.ones(len(client_samples)) / len(client_samples)
    else:
        # 기본 FedAvg: 데이터 크기에 비례한 가중치
        weights = np.array(client_samples) / np.sum(client_samples) if np.sum(client_samples) > 0 else np.ones(len(client_samples)) / len(client_samples)

    # 새 글로벌 모델 상태 계산
    for key in global_dict.keys():
        weighted_sum = torch.zeros_like(global_dict[key])
        for i, state in enumerate(client_states):
            weighted_sum += weights[i] * state[key]

        # 서버 모멘텀 적용
        if use_server_momentum and server_velocity is not None:
            if key not in server_velocity:
                server_velocity[key] = torch.zeros_like(weighted_sum)

            # 현재 업데이트
            update = weighted_sum - global_dict[key]

            # 모멘텀 업데이트
            server_velocity[key] = momentum_beta * server_velocity[key] + (1 - momentum_beta) * update
            global_dict[key] += server_velocity[key]
        else:
            global_dict[key] = weighted_sum

    global_model.load_state_dict(global_dict)
    return global_model, weights

# 3. 통신 최적화 함수들
def calculate_model_size(model):
    """모델 파라미터 크기를 바이트 단위로 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    size_bytes = total_params * 4  # float32 = 4 bytes
    return total_params, size_bytes

def mixed_precision_quantize(weights, key):
    """혼합 정밀도 양자화: 컨볼루션과 완전연결 레이어에 다른 비트 수 적용"""
    if not use_quantization:
        return weights

    def quantize(w, bits):
        min_w = w.min()
        max_w = w.max()
        scale = max_w - min_w
        if scale == 0:
            return w
        norm_w = (w - min_w) / scale
        levels = 2**bits - 1
        quant_w = torch.round(norm_w * levels) / levels * scale + min_w
        return quant_w

    # 컨볼루션 레이어: 낮은 비트 수 적용
    if 'conv' in key or 'features' in key:
        return quantize(weights, quantization_bits_conv)
    # 완전연결 레이어: 높은 비트 수 적용
    elif 'fc' in key or 'classifier' in key or 'linear' in key:
        return quantize(weights, quantization_bits_fc)
    else:
        return weights

def adaptive_pruning_threshold(base_threshold, round_idx, total_rounds):
    """적응적 가지치기: 훈련 초기에는 적게, 후기에는 많이 가지치기"""
    if not use_adaptive_pruning:
        return base_threshold

    # 훈련 초기에는 적게 가지치기, 후반부에 가지치기 비율 증가
    progress = round_idx / total_rounds
    adaptive_threshold = base_threshold * (0.5 + 0.5 * progress)

    return adaptive_threshold

# 4. 데이터셋 준비 함수
def prepare_dataset():
    """CIFAR-10 데이터셋 준비 및 Non-IID 분할"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Non-IID 데이터 분할
    client_indices = partition_dataset(trainset, num_clients, alpha)

    # 클라이언트별 데이터 로더 생성
    client_loaders = []
    for i in range(num_clients):
        indices = client_indices[i]
        subset = Subset(trainset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    # 클라이언트 데이터 특성 계산
    client_data_stats = calculate_client_data_stats(client_loaders, trainset)

    return client_loaders, testloader, client_data_stats

def partition_dataset(dataset, num_clients, alpha):
    """디리클레 분포를 사용한 Non-IID 데이터 분할"""
    num_classes = 10
    targets = np.array(dataset.targets)
    client_indices = {i: [] for i in range(num_clients)}
    class_indices = [np.where(targets == k)[0] for k in range(num_classes)]

    for k, idx_k in enumerate(class_indices):
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet([alpha] * num_clients)
        bounds = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, bounds)
        for i, split in enumerate(splits):
            client_indices[i] += split.tolist()

    for i in client_indices:
        np.random.shuffle(client_indices[i])

    # 클라이언트별 데이터 분포 시각화
    visualize_data_distribution(dataset, client_indices, num_classes, alpha)

    return client_indices

def calculate_client_data_stats(client_loaders, dataset):
    """클라이언트 데이터 특성 계산 및 클러스터링"""
    means = []
    for loader in client_loaders:
        client_data = []
        for x, _ in loader:
            client_data.append(x.to('cpu'))
        if client_data:
            means.append(torch.cat(client_data, dim=0).mean(dim=(0, 2, 3)).numpy())

    if means:
        means = np.stack(means)
        # 클러스터링: 유사한 데이터 분포를 가진 클라이언트 그룹화
        dist_matrix = pairwise_distances(means)
        theta = 0.3  # 클러스터링 임계값
        clusters, assigned = [], set()
        for i in range(len(client_loaders)):
            if i in assigned: continue
            group = []
            for j in range(len(client_loaders)):
                if j not in assigned and dist_matrix[i, j] < theta:
                    group.append(j)
            if group:
                clusters.append(group)
                assigned.update(group)

        if not assigned:  # 클러스터가 생성되지 않았다면
            clusters = [[i] for i in range(len(client_loaders))]

        print("클라이언트 클러스터:", clusters)
        return {"means": means, "clusters": clusters}

    # 기본값 반환
    return {"means": np.zeros((len(client_loaders), 3)),
            "clusters": [[i] for i in range(len(client_loaders))]}

def visualize_data_distribution(dataset, client_indices, num_classes, alpha):
    """클라이언트별 데이터 분포 시각화"""
    class_distribution = np.zeros((num_clients, num_classes))
    for client_id, indices in client_indices.items():
        client_targets = [dataset.targets[idx] for idx in indices]
        for class_id in range(num_classes):
            class_distribution[client_id, class_id] = client_targets.count(class_id)
        # 비율로 정규화
        class_distribution[client_id] = class_distribution[client_id] / len(indices)

    plt.figure(figsize=(12, 6))
    for client_id in range(num_clients):
        plt.bar(np.arange(num_classes) + 0.1*client_id, class_distribution[client_id], width=0.1, label=f'Client {client_id}')
    plt.xlabel('Class')
    plt.ylabel('Data Ratio')
    plt.title(f'Non-IID Data Distribution (α={alpha})')
    plt.legend()
    plt.xticks(range(num_classes))
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('data_distribution.png')

# 6. 비동기 병렬 클라이언트 학습 함수 제거
# async_client_train 함수 삭제

# 7. 서버 업데이트 함수
def server_aggregate(global_model, client_states, client_losses, client_samples, server_velocity=None, momentum_beta=0.9):
    """클라이언트 모델 집계 및 서버 모델 업데이트"""
    global_dict = global_model.state_dict()

    if use_importance_sampling:
        # 손실 기반 가중치 계산 (높은 손실 = 더 많은 가중치)
        weights = np.array([loss ** q for loss in client_losses])
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
        else:
            # 가중치를 계산할 수 없는 경우 샘플 크기 비례로 가중치 설정
            weights = np.array(client_samples) / np.sum(client_samples) if np.sum(client_samples) > 0 else np.ones(len(client_samples)) / len(client_samples)
    else:
        # 기본 FedAvg: 데이터 크기에 비례한 가중치
        weights = np.array(client_samples) / np.sum(client_samples) if np.sum(client_samples) > 0 else np.ones(len(client_samples)) / len(client_samples)

    # 새 글로벌 모델 상태 계산
    for key in global_dict.keys():
        weighted_sum = torch.zeros_like(global_dict[key])
        for i, state in enumerate(client_states):
            weighted_sum += weights[i] * state[key]

        # 서버 모멘텀 적용
        if use_server_momentum and server_velocity is not None:
            if key not in server_velocity:
                server_velocity[key] = torch.zeros_like(weighted_sum)

            # 현재 업데이트
            update = weighted_sum - global_dict[key]

            # 모멘텀 업데이트
            server_velocity[key] = momentum_beta * server_velocity[key] + (1 - momentum_beta) * update
            global_dict[key] += server_velocity[key]
        else:
            global_dict[key] = weighted_sum

    global_model.load_state_dict(global_dict)
    return global_model, weights

# 8. 모델 평가 함수
def evaluate_model(model, testloader):
    """모델 정확도 평가"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            pred = outputs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

# 9. 통합 연합 학습 함수
def federated_learning():
    """향상된 연합 학습 프레임워크 실행 함수"""
    try:
        # 데이터셋 준비
        client_loaders, testloader, client_data_stats = prepare_dataset()
        clusters = client_data_stats["clusters"]

        # 모델 초기화
        global_model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()

        # 모델 크기 계산
        num_params, model_size_bytes = calculate_model_size(global_model)
        model_size_mb = model_size_bytes / (1024 * 1024)
        logging.info(f"모델 파라미터 수: {num_params:,}")
        logging.info(f"모델 크기: {model_size_bytes:,} bytes ({model_size_mb:.2f} MB)")

        # 모니터링 지표 초기화
        accuracies = []
        communication_costs = []
        pruned_communication_costs = []
        client_importance = np.ones(num_clients) / num_clients
        fairness_scores = []
        cluster_losses_per_round = [[] for _ in range(len(clusters))]

        # 서버 모멘텀 초기화
        server_velocity = OrderedDict() if use_server_momentum else None

        # 통신 비용 및 메모리 사용량 모니터링
        total_bytes_transferred = 0
        total_bytes_transferred_optimized = 0

        # 메인 학습 루프
        for r in range(num_rounds):
            logging.info(f"\n===== 라운드 {r+1}/{num_rounds} =====")

            # 활성 클라이언트 선택
            participating_clients = list(range(num_clients))
            if use_active_client_selection and r > 0:
                num_active = max(2, num_clients // 2)
                participating_clients = np.random.choice(
                    num_clients,
                    size=num_active,
                    replace=False,
                    p=client_importance
                ).tolist()

            logging.info(f"참여 클라이언트: {participating_clients}")

            # 라운드별 통신 비용 계산
            round_bytes = 0
            round_bytes_optimized = 0

            # 다운로드 단계: 서버 → 클라이언트 (글로벌 모델 배포)
            download_size = model_size_bytes * len(participating_clients)
            round_bytes += download_size

            # 적응적 가지치기 임계값 계산
            current_pruning_thr = adaptive_pruning_threshold(pruning_thr, r, num_rounds)

            # 최적화된 다운로드 크기 계산 (가지치기 + 양자화)
            compression_ratio = 1.0
            if use_adaptive_pruning:
                compression_ratio *= (1 - current_pruning_thr)

            if use_quantization:
                # 간소화된 계산: 컨볼루션(8비트)과 완전연결(16비트) 레이어의 압축률 평균
                avg_quantization_bits = (quantization_bits_conv + quantization_bits_fc) / 2
                compression_ratio *= (avg_quantization_bits / 32)

            optimized_download_size = model_size_bytes * compression_ratio * len(participating_clients)
            round_bytes_optimized += optimized_download_size

            # 클라이언트 학습 결과 저장용 변수
            client_states = []
            client_losses = []
            client_sample_sizes = []

            # 순차적 클라이언트 학습
            for cid in participating_clients:
                try:
                    client_model = SimpleCNN().to(device)
                    client_model.load_state_dict(global_model.state_dict())
                    optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)

                    updated_model, loss, samples = client_update(
                        client_model,
                        client_loaders[cid],
                        optimizer,
                        criterion,
                        global_model,
                        r,
                        num_rounds,
                        current_pruning_thr,
                        kd_alpha
                    )

                    # 클라이언트 체크포인트 저장
                    save_checkpoint(updated_model, optimizer, r, client_id=cid)

                    client_states.append(updated_model.state_dict())
                    client_losses.append(loss)
                    client_sample_sizes.append(samples)

                finally:
                    # 메모리 해제
                    del client_model
                    torch.cuda.empty_cache()

            # 업로드 단계: 클라이언트 → 서버 (로컬 모델 전송)
            upload_size = model_size_bytes * len(participating_clients)
            round_bytes += upload_size

            # 최적화된 업로드 크기
            optimized_upload_size = model_size_bytes * compression_ratio * len(participating_clients)
            round_bytes_optimized += optimized_upload_size

            # 라운드별 통신 비용 합산
            total_bytes_transferred += round_bytes
            total_bytes_transferred_optimized += round_bytes_optimized

            # 현재 라운드 통신 비용 저장
            communication_costs.append(round_bytes)
            pruned_communication_costs.append(round_bytes_optimized)

            # 서버에서 글로벌 모델 업데이트
            global_model, client_weights = server_aggregate(
                global_model,
                client_states,
                client_losses,
                client_sample_sizes,
                server_velocity,
                momentum_beta
            )

            # 글로벌 모델 체크포인트 저장
            save_checkpoint(global_model, None, r, is_global=True)

            # 클라이언트 중요도 업데이트 (다음 라운드 활성 클라이언트 선택에 사용)
            if use_importance_sampling:
                for i, cid in enumerate(participating_clients):
                    client_importance[cid] = 0.8 * client_importance[cid] + 0.2 * client_weights[i]
                # 중요도 정규화
                client_importance = client_importance / client_importance.sum()

            # 클러스터별 평균 손실 계산
            for i, cluster in enumerate(clusters):
                cluster_clients = [c for c in cluster if c in participating_clients]
                if cluster_clients:
                    avg_loss = np.mean([client_losses[participating_clients.index(c)] for c in cluster_clients])
                    cluster_losses_per_round[i].append(avg_loss)
                else:
                    # 이 라운드에 클러스터에서 참여한 클라이언트가 없는 경우
                    if len(cluster_losses_per_round[i]) > 0:
                        # 이전 라운드 손실 유지
                        cluster_losses_per_round[i].append(cluster_losses_per_round[i][-1])
                    else:
                        cluster_losses_per_round[i].append(0)

            # 참여 클라이언트 간 공정성 점수 계산
            if len(participating_clients) > 1:
                fairness = 1 - (max(client_losses) - min(client_losses)) / max(client_losses)
            else:
                fairness = 1.0  # 참여 클라이언트가 1개면 불공정성 없음
            fairness_scores.append(fairness)

            # 모델 평가
            accuracy = evaluate_model(global_model, testloader)
            accuracies.append(accuracy)
            logging.info(f"라운드 {r+1} 정확도: {accuracy:.2f}%")
            logging.info(f"라운드 {r+1} 공정성: {fairness:.3f}")
            logging.info(f"라운드 {r+1} 통신 비용: {round_bytes / (1024 * 1024):.2f} MB")
            logging.info(f"라운드 {r+1} 최적화된 통신 비용: {round_bytes_optimized / (1024 * 1024):.2f} MB")

        # 최종 모델 저장
        torch.save(global_model.state_dict(), 'enhanced_federated_model.pth')
        logging.info("훈련 완료!")
        logging.info(f"최종 정확도: {accuracies[-1]:.2f}%")
        logging.info(f"총 통신 비용: {total_bytes_transferred / (1024 * 1024):.2f} MB")
        logging.info(f"최적화된 총 통신 비용: {total_bytes_transferred_optimized / (1024 * 1024):.2f} MB")

        # 성능 시각화
        visualize_results(
            accuracies,
            cluster_losses_per_round,
            fairness_scores,
            communication_costs,
            pruned_communication_costs,
            client_importance,
            num_rounds
        )

        return global_model, accuracies, client_importance

    except Exception as e:
        logging.error(f"학습 중 오류 발생: {str(e)}")
        raise

# 10. 결과 시각화 함수
def visualize_results(accuracies, cluster_losses, fairness_scores, communication_costs,
                      pruned_communication_costs, client_importance, num_rounds):
    """훈련 결과 시각화"""
    plt.figure(figsize=(20, 15))

    # 1. 정확도 그래프
    plt.subplot(3, 2, 1)
    plt.plot(range(1, num_rounds+1), accuracies, 'b-', marker='o')
    plt.title('Test Accuracy per Round')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.grid()

    # 2. 클러스터별 손실 그래프
    plt.subplot(3, 2, 2)
    for i, losses in enumerate(cluster_losses):
        plt.plot(range(1, num_rounds+1), losses, marker='o', label=f'Cluster {i}')
    plt.title('Cluster Loss per Round')
    plt.xlabel('Round')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid()

    # 3. 공정성 점수 그래프
    plt.subplot(3, 2, 3)
    plt.plot(range(1, num_rounds+1), fairness_scores, 'g-', marker='o')
    plt.title('Fairness Index per Round')
    plt.xlabel('Round')
    plt.ylabel('Fairness Index')
    plt.grid()

    # 4. 통신 비용 그래프
    plt.subplot(3, 2, 4)
    plt.plot(range(1, num_rounds+1), [c / (1024 * 1024) for c in communication_costs], 'r-', marker='o', label='원본')
    plt.plot(range(1, num_rounds+1), [c / (1024 * 1024) for c in pruned_communication_costs], 'g-', marker='o', label='최적화')
    plt.title('Communication Cost per Round')
    plt.xlabel('Round')
    plt.ylabel('Communication Cost (MB)')
    plt.legend()
    plt.grid()

    # 5. 누적 통신 비용 그래프
    plt.subplot(3, 2, 5)
    cum_cost = np.cumsum([c / (1024 * 1024) for c in communication_costs])
    cum_cost_opt = np.cumsum([c / (1024 * 1024) for c in pruned_communication_costs])
    plt.plot(range(1, num_rounds+1), cum_cost, 'r-', marker='o', label='원본')
    plt.plot(range(1, num_rounds+1), cum_cost_opt, 'g-', marker='o', label='최적화')
    plt.title('Cumulative Communication Cost')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Cost (MB)')
    plt.legend()
    plt.grid()

    # 6. 클라이언트 중요도 그래프
    plt.subplot(3, 2, 6)
    plt.bar(range(len(client_importance)), client_importance)
    plt.title('Final Client Importance')
    plt.xlabel('Client ID')
    plt.ylabel('Importance Weight')
    plt.xticks(range(len(client_importance)))
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig('enhanced_federated_learning_results.png')
    plt.close()

# 11. 하이퍼파라미터 최적화 함수
def hyperparameter_optimization(param_grid, n_trials=5):
    """간단한 그리드 서치를 통한 하이퍼파라미터 최적화"""
    best_accuracy = 0
    best_params = None
    results = []

    # 가능한 모든 하이퍼파라미터 조합 생성
    import itertools
    param_combinations = list(itertools.product(*param_grid.values()))
    param_keys = list(param_grid.keys())

    print(f"총 {len(param_combinations)} 개의 하이퍼파라미터 조합 시험")

    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_keys, params))
        print(f"\n시험 {i+1}/{len(param_combinations)}: {param_dict}")

        # 하이퍼파라미터 설정
        global q, alpha, pruning_thr, momentum_beta, kd_alpha
        global use_adaptive_pruning, use_importance_sampling, use_knowledge_distillation
        global use_quantization, use_server_momentum, use_active_client_selection

        q = param_dict.get('q', q)
        pruning_thr = param_dict.get('pruning_thr', pruning_thr)
        momentum_beta = param_dict.get('momentum_beta', momentum_beta)
        kd_alpha = param_dict.get('kd_alpha', kd_alpha)

        use_adaptive_pruning = param_dict.get('use_adaptive_pruning', use_adaptive_pruning)
        use_importance_sampling = param_dict.get('use_importance_sampling', use_importance_sampling)
        use_knowledge_distillation = param_dict.get('use_knowledge_distillation', use_knowledge_distillation)
        use_server_momentum = param_dict.get('use_server_momentum', use_server_momentum)

        # n_trials 횟수만큼 다른 시드로 실행하여 평균 성능 계산
        trial_accuracies = []
        for trial in range(n_trials):
            print(f"시도 {trial+1}/{n_trials}")
            torch.manual_seed(trial)
            np.random.seed(trial)

            # 축소된 라운드로 빠른 평가
            global num_rounds
            original_rounds = num_rounds
            num_rounds = 20  # 빠른 평가를 위해 라운드 수 축소

            _, accuracies, _ = federated_learning()
            trial_accuracies.append(accuracies[-1])  # 최종 정확도 저장

            # 원래 라운드 수 복원
            num_rounds = original_rounds

        # 평균 정확도 계산
        mean_accuracy = np.mean(trial_accuracies)
        std_accuracy = np.std(trial_accuracies)

        print(f"평균 정확도: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")

        # 결과 저장
        results.append({
            'params': param_dict,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'trial_accuracies': trial_accuracies
        })

        # 최고 성능 파라미터 업데이트
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = param_dict

    # 결과 정렬 및 출력
    results.sort(key=lambda x: x['mean_accuracy'], reverse=True)

    print("\n===== 하이퍼파라미터 최적화 결과 =====")
    print(f"최고 정확도: {best_accuracy:.2f}%")
    print(f"최적 파라미터: {best_params}")

    print("\n상위 3개 파라미터 설정:")
    for i, result in enumerate(results[:3]):
        print(f"{i+1}. 정확도: {result['mean_accuracy']:.2f}% ± {result['std_accuracy']:.2f}%")
        print(f"   파라미터: {result['params']}")

    # 최적 파라미터로 설정
    q = best_params.get('q', q)
    pruning_thr = best_params.get('pruning_thr', pruning_thr)
    momentum_beta = best_params.get('momentum_beta', momentum_beta)
    kd_alpha = best_params.get('kd_alpha', kd_alpha)

    use_adaptive_pruning = best_params.get('use_adaptive_pruning', use_adaptive_pruning)
    use_importance_sampling = best_params.get('use_importance_sampling', use_importance_sampling)
    use_knowledge_distillation = best_params.get('use_knowledge_distillation', use_knowledge_distillation)
    use_server_momentum = best_params.get('use_server_momentum', use_server_momentum)

    return best_params, results

def save_checkpoint(model, optimizer, epoch, client_id=None, is_global=False):
    """체크포인트 저장 함수"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
    }
    
    if is_global:
        filename = f'global_model_epoch_{epoch}.pth'
    else:
        filename = f'client_{client_id}_epoch_{epoch}.pth'
    
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(checkpoint, path)
    logging.info(f'체크포인트 저장됨: {path}')

def load_checkpoint(model, optimizer, path):
    """체크포인트 로드 함수"""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f'체크포인트 로드됨: {path}')
        return checkpoint['epoch']
    return 0

# 메인 실행
if __name__ == "__main__":
    print("=== 향상된 비동기 연합 학습 프레임워크 ===")

    # 하이퍼파라미터 최적화 여부
    run_optimization = False

    if run_optimization:
        # 하이퍼파라미터 최적화 실행
        param_grid = {
            'q': [1.0, 1.2, 1.5],
            'pruning_thr': [0.05, 0.1, 0.15],
            'momentum_beta': [0.8, 0.9, 0.95],
            'kd_alpha': [0.3, 0.5, 0.7],
            'use_adaptive_pruning': [True],
            'use_importance_sampling': [True, False],
            'use_knowledge_distillation': [True, False],
            'use_server_momentum': [True, False]
        }

        best_params, _ = hyperparameter_optimization(param_grid, n_trials=3)
        print(f"최적 파라미터 적용 후 훈련 실행...")

    # 메인 연합 학습 실행
    global_model, accuracies, client_importance = federated_learning()

    print("훈련 완료!")
    print(f"최종 정확도: {accuracies[-1]:.2f}%")