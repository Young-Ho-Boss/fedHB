import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# 1. 설정
num_clients = 5
batch_size = 64
num_rounds = 150
local_epochs = 5
alpha = 0.5 # Non-IID 분할

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.manual_seed(0)
np.random.seed(0)

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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 3. 데이터셋 분할
def partition_dataset(dataset, num_clients, alpha):
    num_classes = 10
    targets = np.array(dataset.targets)
    c_indices = {i: [] for i in range(num_clients)}
    class_indices = [np.where(targets==k)[0] for k in range(num_classes)]
    for k, idx_k in enumerate(class_indices):
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet([alpha]*num_clients)
        bounds = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
        splits = np.split(idx_k, bounds)
        for i, split in enumerate(splits):
            c_indices[i] += split.tolist()
    for i in c_indices:
        np.random.shuffle(c_indices[i])
    visualize_data_distribution(dataset, c_indices, num_classes, alpha)
    return c_indices

def visualize_data_distribution(dataset, c_indices, num_classes, alpha):
    class_distribution = np.zeros((num_clients, num_classes))
    for client_id, indices in c_indices.items():
        client_targets = [dataset.targets[idx] for idx in indices]
        for class_id in range(num_classes):
            class_distribution[client_id, class_id] = client_targets.count(class_id)
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

def prepare_dataset():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    train = datasets.CIFAR10('./data', True, tf, download=True)
    test  = datasets.CIFAR10('./data', False, tf, download=True)

    idxs = partition_dataset(train, num_clients, alpha)
    c_loaders = []
    for i in range(num_clients):
        subset = Subset(train, idxs[i])
        c_loaders.append(DataLoader(subset, batch_size, shuffle=True))
    test_loader = DataLoader(test, 128, shuffle=False)
    return c_loaders, test_loader

# 4. 로컬 업데이트 (FedAvg)
def client_update_fedavg(c_model, data_loader, criterion, optimizer):
    c_model.train()
    for _ in range(local_epochs):
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(c_model(x), y)
            loss.backward()
            optimizer.step()
    return c_model.state_dict(), len(data_loader.dataset)

# 5. 서버 집계
def server_aggregate(g_model, c_states, c_samples):
    global_dict = g_model.state_dict()
    total = sum(c_samples)
    for k in global_dict:
        global_dict[k] = sum(c_states[i][k]*c_samples[i] for i in range(len(c_states))) / total
    g_model.load_state_dict(global_dict)
    return g_model

# 6. 평가
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            predict = model(x).argmax(dim=1)
            correct += (predict==y).sum().item()
            total += y.size(0)
    return 100*correct/total if total > 0 else 0

# 7. 실행
def run_fedavg():
    print("=== FedAvg Test ===")

    # 데이터셋 준비
    c_loaders, test_loader = prepare_dataset()

    # 모델 초기화
    global_model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # 모델 크기 계산
    num_params = sum(p.numel() for p in global_model.parameters())
    model_size_bytes = num_params * 4
    model_size_mb = model_size_bytes / (1024 * 1024)
    print(f"model params count: {num_params:,}")
    print(f"model size: {model_size_bytes:,} bytes ({model_size_mb:.2f} MB)")

    # 메인 학습 루프
    initial_start = time.time()
    accs = []
    round_times = []
    for r in range(1, num_rounds+1):
        round_start = time.time()
        print(f"[FedAvg] Round {r}/{num_rounds}", end="")
        states, samples = [], []
        for c_loader in c_loaders:
            lm = SimpleCNN().to(device)
            lm.load_state_dict(global_model.state_dict())
            opt = optim.SGD(lm.parameters(), lr=0.01)
            state, cnt = client_update_fedavg(lm, c_loader, criterion, opt)
            states.append(state); samples.append(cnt)

        global_model = server_aggregate(global_model, states, samples)
        acc = evaluate(global_model, test_loader)
        accs.append(acc)
        round_end = time.time()
        round_times.append(round_end - round_start)
        print(f"- Accuracy: {acc:.2f}% | Completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(round_end))}")

    plt.plot(range(1,num_rounds+1), accs, marker='o')
    plt.xlabel('Round'); plt.ylabel('Accuracy (%)')
    plt.title('FedAvg Accuracy')
    plt.grid(alpha=0.3)
    plt.show()

    total_time = time.time() - initial_start
    avg_time = total_time / num_rounds

    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(dev)
        props = torch.cuda.get_device_properties(dev)
        gpu_mem = props.total_memory / (1024**3)  # GB 단위
    else:
        gpu_name = "CPU"
        gpu_mem = 0

    print("=== Test Complete ===")
    print (f"final Accuracy      : {acc:.2f}%")
    print(f"Total Time           : {total_time:.2f} sec")
    print(f"Round Time (Average) : {avg_time:.2f} sec")
    print(f"Device               : {gpu_name} ({gpu_mem:.1f} GB)")

if __name__ == '__main__':
    run_fedavg()
