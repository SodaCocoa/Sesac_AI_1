# best code
import numpy as np
import os
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

pathFolder = "../mod/"
os.makedirs(pathFolder, exist_ok=True)
xTrainName = "xTrain1.pkl"
yTrainName = "yTrain_onehot.pkl"

# 데이터 로드
with open(pathFolder + xTrainName, 'rb') as f1:
    X = pickle.load(f1)

with open(pathFolder + yTrainName, 'rb') as f2:
    y = pickle.load(f2)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return data, torch.tensor(self.labels[idx], dtype=torch.float32)

class FCNet(nn.Module):
    def __init__(self, input_features):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_features, 128)  # Adjust the number of input features accordingly
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Assuming binary classification

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, train_loader, device, optimizer, criterion, epochs):
    model.train()
    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_loss = 0.0
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

def evaluate_model(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels_max = torch.max(labels, 1)  # one-hot 인코딩된 레이블에서 최대값을 가진 인덱스를 찾습니다.
            total += labels.size(0)
            correct += (predicted == labels_max).sum().item()  # 예측된 인덱스와 레이블의 인덱스를 비교합니다.
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return accuracy, avg_loss


# 메인 실행 블록
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rates = [0.0005, 0.00025,0.00001]
    batch_sizes = [4, 8,16,32,64]
    epochs = 200

    best_accuracy = 0.0
    best_loss = float('inf')
    best_parameters = {}

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Training with LR={lr}, Batch Size={batch_size}")
            model = FCNet(input_features=X_train.shape[1]).to(device)  # 평탄화된 입력에 맞게 조정
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            train_loader = DataLoader(CustomDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(CustomDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
            train_model(model, train_loader, device, optimizer, criterion, epochs)
            val_accuracy, val_loss = evaluate_model(model, val_loader, device, criterion)
            print(f'Validation - Accuracy: {val_accuracy:.2f}%, Loss: {val_loss:.4f}')

            if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
                best_accuracy = val_accuracy
                best_loss = val_loss
                best_parameters = {'learning_rate': lr, 'batch_size': batch_size, 'loss': val_loss}
                torch.save(model.state_dict(), f'best_model_{lr}_{batch_size}.pth')

    print(f"Best Parameters: {best_parameters}")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%, Loss: {best_parameters['loss']:.4f}")




#     Best Parameters: {'learning_rate': 0.0005, 'batch_size': 4, 'loss': 0.5286541449924698}
# Best Validation Accuracy: 83.12%, Loss: 0.5287