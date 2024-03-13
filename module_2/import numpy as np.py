import numpy as np
import math
import pickle
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # 데이터를 (1, length) 형태로 변환
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return data, torch.tensor(self.labels[idx], dtype=torch.long)


# 모델 정의 시 입력 데이터 형태를 고려하여 Conv1d 계층 설정
class Conv1DNet(nn.Module):
    def __init__(self, num_features):
        super(Conv1DNet, self).__init__()
        # 예시: 입력 채널을 1로 설정
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Conv1D 계층을 거친 후의 텐서 크기에 맞게 Linear 계층 입력 조정
        self.fc1 = nn.Linear(32 * num_features, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 평탄화
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Conv1DNet(nn.Module):
    def __init__(self, num_features):
        super(Conv1DNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * num_features, 64)
        self.fc2 = nn.Linear(64, 2)  # 2개 클래스 분류

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 평탄화
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    pathFolder = "./wine_quality/"
    xTrainName = "xTrain1.pkl"
    yTrainName = "yTrain.pkl"

    with open(pathFolder + xTrainName, 'rb') as f1:
        X = pickle.load(f1)

    with open(pathFolder + yTrainName, 'rb') as f2:
        y = pickle.load(f2)

    X_train, X_tv, y_train, y_tv = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tv, y_tv, test_size=0.5, random_state=42)

    # transform = transforms.Compose([
    #     lambda x: make_imglike(x, target_size=X.shape[1])
    # ])

    train_dataset = CustomDataset(X_train, y_train )
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Conv1DNet(num_features=X.shape[1]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}%')
