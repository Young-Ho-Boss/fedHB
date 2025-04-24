import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics.pairwise import pairwise_distances
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 설정
num_clients = 5
num_rounds = 30
local_epochs = 5   # local 학습량 증가
q = 0.5
alpha = 0.5       # 비균형 완화 IID 1.0 / 중간 0.5 극악 0.1
batch_size = 64
mu = 0.02
prune_ratio = 0.05
prune_start_round = 10  # pruning 시작 라운드

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("PyTorch 버전:", torch.__version__)
print("CUDA 사용 가능:", torch.cuda.is_available())
print("CUDA 버전:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU 이름:", torch.cuda.get_device_name(0))

    # GPU 메모리 테스트
    try:
        x = torch.rand(1000, 1000).cuda()
        print("GPU 메모리에 텐서 생성 완료")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print("GPU 메모리 사용 중 오류:", e)
else:
    print("GPU를 사용할 수 없습니다.")
print(f"Using device: {device}")
torch.manual_seed(0)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.net(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

def partition_dataset(dataset, num_clients, alpha):
    targets = np.array(dataset.targets)
    client_indices = {i: [] for i in range(num_clients)}
    class_indices = [np.where(targets == k)[0] for k in range(10)]
    for k, idx_k in enumerate(class_indices):
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet([alpha] * num_clients)
        bounds = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, bounds)
        for i, split in enumerate(splits):
            client_indices[i] += split.tolist()
    for i in client_indices:
        np.random.shuffle(client_indices[i])
    return client_indices

client_splits = partition_dataset(trainset, num_clients, alpha)
client_loaders = []
for i in range(num_clients):
    subset = Subset(trainset, client_splits[i])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    client_loaders.append(loader)

# 클러스터링
means = []
for loader in client_loaders:
    client_data = []
    for x, _ in loader:
        client_data.append(x.to('cpu'))
    means.append(torch.cat(client_data, dim=0).mean(dim=(0, 2, 3)).numpy())
means = np.stack(means)

dist_matrix = pairwise_distances(means)
theta = 0.3
clusters, assigned = [], set()
for i in range(num_clients):
    if i in assigned: continue
    group = []
    for j in range(num_clients):
        if j not in assigned and dist_matrix[i, j] < theta:
            group.append(j)
    if group:
        clusters.append(group)
        assigned.update(group)
print("Clusters:", clusters)

# 모델 준비
global_model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
testloader = DataLoader(testset, batch_size=128, shuffle=False)

accuracies, fairness_scores = [], []
cluster_losses_per_round = [[] for _ in range(len(clusters))]
# 학습 루프
for r in range(num_rounds):
    client_states, client_losses = [], []

    # 클라이언트 학습
    for cid in range(num_clients):
        model = SimpleCNN().to(device)
        model.load_state_dict(global_model.state_dict())
        opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        total_loss, total_samples = 0.0, 0
        for _ in range(local_epochs):
            for x, y in client_loaders[cid]:
                # 데이터와 레이블을 동시에 device로 이동
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)

                # proximal term 계산
                prox = sum(torch.norm(w - w0)**2 for w, w0 in zip(model.parameters(), global_model.parameters()))
                total = loss + (mu / 2) * prox
                total.backward()
                opt.step()
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

        client_states.append(model.state_dict())
        client_losses.append(total_loss / total_samples)

    # q-FedAvg: 손실 기반 클라이언트 선택 및 가중치 계산
    losses_q = [l**(q+1) for l in client_losses]
    sum_q = sum(l**q for l in client_losses)
    weights = [lq / sum_q for lq in losses_q]

    # 새 글로벌 모델 상태 업데이트
    new_state = {}
    for key in global_model.state_dict().keys():
        weighted_sum = sum(weights[c] * client_states[c][key] for c in range(num_clients))
        new_state[key] = weighted_sum
    global_model.load_state_dict(new_state)

    # Pruning: prune_start_round 이후부터 적용
    if r >= prune_start_round:
        # 가중치의 절대값 크기 순으로 정렬
        all_weights = torch.cat([p.view(-1).abs() for p in global_model.parameters()])
        threshold = torch.quantile(all_weights, prune_ratio)  # 하위 prune_ratio 만큼의 가중치 제거
        with torch.no_grad():
            for p in global_model.parameters():
                mask = p.abs() >= threshold
                p.mul_(mask)  # 임계값 이하의 가중치를 0으로 설정

    # 클러스터별 평균 손실 계산
    for i, cluster in enumerate(clusters):
        avg_loss = np.mean([client_losses[c] for c in cluster])
        cluster_losses_per_round[i].append(avg_loss)

    # 공정성 점수 계산
    fairness = 1 - (max(client_losses) - min(client_losses)) / max(client_losses)
    fairness_scores.append(fairness)

    # 모델 평가
    correct, total = 0, 0
    global_model.eval()
    with torch.no_grad():
        for x, y in testloader:
            # 평가 데이터도 device로 이동
            x, y = x.to(device), y.to(device)
            outputs = global_model(x)
            pred = outputs.argmax(dim=1)
            # 이제 pred와 y 모두 같은 device에 있음
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total
    accuracies.append(acc)

    print(f"Round {r+1}/{num_rounds}: Accuracy={acc:.2f}%, Fairness={fairness:.3f}")
    # 5라운드마다 자세한 정보 출력
    if (r+1) % 5 == 0:
        print(f"  Client losses: {np.round(client_losses, 3)}")
        print(f"  Client weights: {np.round(weights, 3)}")

        # 모델 가중치의 희소성 평가 (0인 가중치의 비율)
        total_params = sum(p.numel() for p in global_model.parameters())
        zero_params = sum((p == 0).sum().item() for p in global_model.parameters())
        sparsity = 100 * zero_params / total_params
        print(f"  Model sparsity: {sparsity:.2f}%")

print("훈련 완료!")

# 결과 시각화
plt.figure(figsize=(15, 5))

# 1. 정확도 그래프
plt.subplot(1, 3, 1)
plt.plot(accuracies)
plt.title('Test Accuracy per Round')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.grid()

# 2. 클러스터별 손실 그래프
plt.subplot(1, 3, 2)
for i, losses in enumerate(cluster_losses_per_round):
    plt.plot(losses, label=f'Cluster {i+1}')
plt.title('Cluster Loss per Round')
plt.xlabel('Round')
plt.ylabel('Average Loss')
plt.legend()
plt.grid()

# 3. 공정성 점수 그래프
plt.subplot(1, 3, 3)
plt.plot(fairness_scores)
plt.title('Fairness Index per Round')
plt.xlabel('Round')
plt.ylabel('Fairness Index')
plt.grid()

plt.tight_layout()
plt.show()

# 추가 분석: 프루닝 전후 정확도 비교
plt.figure(figsize=(10, 5))
plt.plot(accuracies)
plt.axvline(x=prune_start_round, color='r', linestyle='--', label='Pruning Start')
plt.title('Accuracy with Pruning')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()
plt.show()

# 클라이언트별 최종 손실 비교
plt.figure(figsize=(8, 5))
plt.bar(range(num_clients), client_losses)
plt.title('Final Loss per Client')
plt.xlabel('Client ID')
plt.ylabel('Loss')
plt.xticks(range(num_clients))
plt.grid(axis='y')
plt.show()
