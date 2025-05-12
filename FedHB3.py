import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import random

# ----- 모델 정의 -----
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNet18Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MentorMentee(nn.Module):
    def __init__(self):
        super().__init__()
        self.mentor = ResNet18()
        self.mentee = ResNet18Tiny()

    def forward(self, x):
        with torch.no_grad():
            mentor_features = self.mentor.features(x)
        output = self.mentee(mentor_features)
        return output

# ----- Knowledge Distillation Loss -----
def distillation_loss(student_logits, teacher_logits, targets, T=3.0, alpha=0.7):
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits/T, dim=1),
        F.softmax(teacher_logits/T, dim=1)
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, targets)
    return alpha * soft_loss + (1 - alpha) * hard_loss

# ----- 데이터 분할 -----
def get_client_loaders(num_clients=5, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    data_per_client = len(trainset) // num_clients
    client_loaders = []
    indices = np.random.permutation(len(trainset))
    for i in range(num_clients):
        idx = indices[i*data_per_client:(i+1)*data_per_client]
        client_subset = Subset(trainset, idx)
        loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return client_loaders, test_loader

# ----- 클라이언트 로컬 학습 -----
def local_train(model, loader, device, epochs=1):
    model.train()
    optimizer = optim.Adam([
        {'params': model.mentor.parameters(), 'lr': 1e-3},
        {'params': model.mentee.parameters(), 'lr': 1e-3}
    ])
    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            mentor_logits = model.mentor(data)
            with torch.no_grad():
                mentor_features = model.mentor.features(data)
            mentee_logits = model.mentee(mentor_features)
            loss = distillation_loss(mentee_logits, mentor_logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# ----- 통신: 멘티 파라미터만 -----
def get_mentee_state_dict(model):
    return copy.deepcopy(model.mentee.state_dict())

def set_mentee_state_dict(model, mentee_state_dict):
    model.mentee.load_state_dict(mentee_state_dict)

# ----- 서버 집계 -----
def server_aggregate(mentee_state_dicts):
    avg_state_dict = copy.deepcopy(mentee_state_dicts[0])
    for key in avg_state_dict.keys():
        for i in range(1, len(mentee_state_dicts)):
            avg_state_dict[key] += mentee_state_dicts[i][key]
        avg_state_dict[key] = avg_state_dict[key] / len(mentee_state_dicts)
    return avg_state_dict

# ----- 평가 -----
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            mentor_features = model.mentor.features(data)
            output = model.mentee(mentor_features)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

# ----- 전체 연합학습 루프 -----
def federated_learning(num_clients=5, rounds=10, local_epochs=1, device='cpu'):
    client_loaders, test_loader = get_client_loaders(num_clients)
    clients = [MentorMentee().to(device) for _ in range(num_clients)]
    global_mentee_state = None

    for rnd in range(rounds):
        mentee_state_dicts = []
        for i, client in enumerate(clients):
            if global_mentee_state is not None:
                set_mentee_state_dict(client, global_mentee_state)
            local_train(client, client_loaders[i], device, epochs=local_epochs)
            mentee_state_dicts.append(get_mentee_state_dict(client))
        global_mentee_state = server_aggregate(mentee_state_dicts)
        # (선택) 매 라운드 평가
        set_mentee_state_dict(clients[0], global_mentee_state)
        acc = evaluate(clients[0], test_loader, device)
        print(f"Round {rnd+1} - Test Accuracy: {acc:.4f}")

    # 최종 평가
    set_mentee_state_dict(clients[0], global_mentee_state)
    acc = evaluate(clients[0], test_loader, device)
    print(f"Final Test Accuracy: {acc:.4f}")

# ----- 실행 -----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    federated_learning(num_clients=5, rounds=10, local_epochs=1, device=device)
