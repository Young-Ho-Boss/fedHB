import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ----------------------------
# 1. 모델 정의 (SimpleCNN)
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ----------------------------
# 2. 모델 압축 기법
# ----------------------------
class ModelQuantizer:
    def __init__(self, bits=8):
        self.bits = bits
        self.scale_factor = 2**(bits - 1) - 1

    def quantize_model(self, state_dict):
        quantized = {}
        scales = {}
        for name, tensor in state_dict.items():
            if tensor.dtype == torch.float32:
                mn, mx = tensor.min(), tensor.max()
                scale = (mx - mn) / (2 * self.scale_factor)
                scales[name] = (mn, scale)
                q = torch.round((tensor - mn) / scale - self.scale_factor).to(torch.int8)
                quantized[name] = q
            else:
                quantized[name] = tensor
        return quantized, scales

    def dequantize_model(self, quantized, scales):
        dequantized = {}
        for name, q in quantized.items():
            if name in scales:
                mn, scale = scales[name]
                dequantized[name] = (q.float() + self.scale_factor) * scale + mn
            else:
                dequantized[name] = q
        return dequantized

def top_k_sparsification(state_dict, k_ratio=0.1):
    sparse, idxs = {}, {}
    for name, tensor in state_dict.items():
        flat = tensor.flatten()
        k = max(1, int(len(flat) * k_ratio))
        _, top_idx = torch.topk(flat.abs(), k)
        sparse[name] = flat[top_idx]
        idxs[name] = (top_idx, tensor.shape)
    return sparse, idxs

def reconstruct_from_sparse(sparse, idxs):
    full = {}
    for name, vals in sparse.items():
        idx, shape = idxs[name]
        recon = torch.zeros(np.prod(shape))
        recon[idx] = vals
        full[name] = recon.view(shape)
    return full

# ----------------------------
# 3. ALT: 적응형 로컬 epoch
# ----------------------------
def calculate_representation_similarity(local_model, global_model, data_loader, device):
    local_model.eval(); global_model.eval()
    sims = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            lf = local_model.features(x).view(x.size(0), -1)
            gf = global_model.features(x).view(x.size(0), -1)
            sims.append(F.cosine_similarity(lf, gf, dim=1).mean().item())
    return np.mean(sims)

def adaptive_local_epochs(similarity, round_idx, total_rounds, base=5):
    sf = max(0.5, similarity)
    pf = 1 + (round_idx / total_rounds) * 0.5
    e = int(base * sf * pf)
    return max(1, min(10, e))

# ----------------------------
# 4. ALT 기반 클라이언트 업데이트
# ----------------------------
def alt_client_update(client_model, global_model, data_loader, criterion,
                      round_idx, total_rounds, device):
    sim = calculate_representation_similarity(client_model, global_model, data_loader, device)
    epochs = adaptive_local_epochs(sim, round_idx, total_rounds)
    client_model.train()
    optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
    total_loss = 0.0
    for _ in range(epochs):
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = client_model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)
    return client_model, total_loss / (len(data_loader.dataset) * epochs), epochs, sim

# ----------------------------
# 5. 통신 효율적 FedHB 클래스
# ----------------------------
class CommunicationEfficientFedHB:
    def __init__(self, compression_ratio=0.1, quant_bits=8, sampling_ratio=0.3):
        self.comp = compression_ratio
        self.samp = sampling_ratio
        self.quantizer = ModelQuantizer(bits=quant_bits)

    def efficient_client_update(self, client_model, global_model, loader,
                                criterion, round_idx, total_rounds, device):
        updated, loss, epochs, sim = alt_client_update(
            client_model, global_model, loader, criterion, round_idx, total_rounds, device
        )
        sd = updated.state_dict()
        sparse, idxs = top_k_sparsification(sd, self.comp)
        quantized, scales = self.quantizer.quantize_model(sparse)
        return {'quant': quantized, 'idxs': idxs, 'scales': scales,
                'epochs': epochs, 'sim': sim}

    def server_aggregate(self, global_model, updates):
        # 중요도: (1 - sim) * epochs
        imps = [(i, (1 - u['sim']) * u['epochs']) for i, u in enumerate(updates)]
        imps.sort(key=lambda x: x[1], reverse=True)
        selected = [i for i, _ in imps[:max(1, int(len(updates) * self.samp))]]
        agg, total_w = {}, 0.0
        for idx in selected:
            u = updates[idx]
            sparse = self.quantizer.dequantize_model(u['quant'], u['scales'])
            full = reconstruct_from_sparse(sparse, u['idxs'])
            w = u['epochs']; total_w += w
            for name, param in full.items():
                agg.setdefault(name, torch.zeros_like(param))
                agg[name] += param * w
        for name in agg:
            agg[name] /= total_w
        global_model.load_state_dict(agg)
        return global_model

# ----------------------------
# 6. 데이터 준비 (Non-IID CIFAR-10)
# ----------------------------
def prepare_non_iid_cifar(num_clients, alpha=0.5, batch_size=64):
    trans_train = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    train = datasets.CIFAR10('./data', train=True, download=True, transform=trans_train)
    test  = datasets.CIFAR10('./data', train=False, download=True, transform=trans_test)
    # Dirichlet 분할
    labels = np.array(train.targets)
    client_indices = [[] for _ in range(num_clients)]
    for cls in range(10):
        idx = np.where(labels == cls)[0]
        np.random.shuffle(idx)
        proportions = np.random.dirichlet([alpha]*num_clients)
        proportions = (proportions / proportions.sum() * len(idx)).astype(int)
        ptr = 0
        for c, cnt in enumerate(proportions):
            client_indices[c] += idx[ptr:ptr+cnt].tolist()
            ptr += cnt
    client_loaders = []
    for idxs in client_indices:
        subset = Subset(train, idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        client_loaders.append(loader)
    test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    return client_loaders, test_loader

# ----------------------------
# 7. 메인 연합학습 실행
# ----------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_clients = 10
    num_rounds = 200
    criterion = nn.CrossEntropyLoss()

    client_loaders, test_loader = prepare_non_iid_cifar(num_clients, alpha=0.5)
    global_model = SimpleCNN().to(device)
    fed = CommunicationEfficientFedHB(compression_ratio=0.1, quant_bits=8, sampling_ratio=0.3)

    for r in range(num_rounds):
        updates = []
        comm_cost = 0
        for loader in client_loaders:
            cm = SimpleCNN().to(device)
            cm.load_state_dict(global_model.state_dict())
            u = fed.efficient_client_update(cm, global_model, loader,
                                            criterion, r, num_rounds, device)
            updates.append(u)
            # 통신량 계산 (대략)
            orig = sum(p.numel() for p in cm.parameters()) * 4
            comm = len(u['quant']) * 1 + sum(idx.numel() for idx, _ in u['idxs'].values()) * 4
            comm_cost += comm
        global_model = fed.server_aggregate(global_model, updates)
        # 평가
        global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = global_model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total * 100
        print(f"Round {r+1:03d} | Acc: {acc:.2f}% | Comm: {comm_cost/1024/1024:.2f} MB")
        if acc >= 90.0:
            break

if __name__ == "__main__":
    main()
