#hyukjoo_NN.py
import numpy as np
import os
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """데이터셋 초기화 함수."""
        self.data = data
        self.labels = labels

    def __len__(self):
        """데이터셋의 길이(데이터의 총 개수)를 반환."""
        return len(self.data)

    def __getitem__(self, idx):
        """주어진 인덱스 idx에 해당하는 데이터를 반환."""
        data = self.data[idx]
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return data, labels

class FCNet(nn.Module):
    def __init__(self, input_features):
        """모델 초기화 함수."""
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_features, 128)  # 첫 번째 풀리 커넥티드(완전 연결) 레이어
        self.fc2 = nn.Linear(128, 64)  # 두 번째 풀리 커넥티드 레이어
        self.fc3 = nn.Linear(64, 2)  # 세 번째 풀리 커넥티드 레이어, 이진 분류를 가정

    def forward(self, x):
        """모델의 순방향 패스를 정의."""
        x = x.view(x.size(0), -1)  # 입력을 펼침
        x = torch.relu(self.fc1(x))  # 첫 번째 레이어 후 ReLU 활성화 함수 적용
        x = torch.relu(self.fc2(x))  # 두 번째 레이어 후 ReLU 활성화 함수 적용
        x = self.fc3(x)  # 세 번째 레이어의 출력
        return x

class ModelTrainer:
    def __init__(self, model, device, learning_rate, batch_size, epochs):
        """트레이너 초기화 함수."""
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)  # 옵티마이저 설정
        self.criterion = nn.CrossEntropyLoss()  # 손실 함수 설정

    def train_model(self, train_loader):
        """모델 학습 함수."""
        self.model.train()  # 모델을 학습 모드로 설정
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            running_loss = 0.0
            for data, labels in train_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()  # 옵티마이저의 그래디언트 초기화
                outputs = self.model(data)  # 모델의 순방향 패스
                loss = self.criterion(outputs, labels)  # 손실 계산
                loss.backward()  # 역방향 패스 (그래디언트 계산)
                self.optimizer.step()  # 모델 업데이트
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)  # 평균 손실 계산
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

    def evaluate_model(self, loader):
        """모델 평가 함수."""
        self.model.eval()  # 모델을 평가 모드로 설정
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # 그래디언트 계산 비활성화
            for data, labels in loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, labels_max = torch.max(labels, 1)
                total += labels.size(0)
                correct += (predicted == labels_max).sum().item()
        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / total  # 정확도 계산
        return accuracy, avg_loss

def main():
    pathFolder = "../module_2/wine_quality/"  # 데이터 저장 폴더
    os.makedirs(pathFolder, exist_ok=True)  # 폴더 생성 (이미 존재하면 무시)
    xTrainName = "xTrain1.pkl"  # 학습 데이터 파일명
    yTrainName = "yTrain_onehot.pkl"  # 레이블 데이터 파일명

    # 데이터 로드
    with open(pathFolder + xTrainName, 'rb') as f1:
        X = pickle.load(f1)
    with open(pathFolder + yTrainName, 'rb') as f2:
        y = pickle.load(f2)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)  # 데이터 분할

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 디바이스 설정
    learning_rates = [0.0005, 0.00025,0.00001,0.0001] #학습률
    batch_sizes = [2,4, 8,16,32,64] # 배치크기
    epochs = 200 #에폭수

    best_accuracy = 0.0  # 최고 정확도
    best_loss = float('inf')  # 최저 손실
    best_parameters = {}  # 최적 파라미터

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Training with LR={lr}, Batch Size={batch_size}")
            model = FCNet(input_features=X_train.shape[1]).to(device)  # 모델 초기화
            trainer = ModelTrainer(model, device, lr, batch_size, epochs)  # 트레이너 초기화
            train_loader = DataLoader(CustomDataset(X_train, y_train), batch_size=batch_size, shuffle=True)  # 훈련 데이터 로더
            val_loader = DataLoader(CustomDataset(X_val, y_val), batch_size=batch_size, shuffle=False)  # 검증 데이터 로더
            trainer.train_model(train_loader)  # 모델 학습
            val_accuracy, val_loss = trainer.evaluate_model(val_loader)  # 모델 평가
            print(f'Validation - Accuracy: {val_accuracy:.2f}%, Loss: {val_loss:.4f}')

            if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
                best_accuracy = val_accuracy
                best_loss = val_loss
                best_parameters = {'learning_rate': lr, 'batch_size': batch_size, 'loss': val_loss}
                torch.save(model.state_dict(), f'best_model_{lr}_{batch_size}.pth')  # 최적 모델 저장

    print(f"Best Parameters: {best_parameters}")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%, Loss: {best_parameters['loss']:.4f}")

if __name__ == '__main__':
    main()
# Best Parameters: {'learning_rate': 0.0005, 'batch_size': 32, 'loss': 0.45122678875923156}
# Best Validation Accuracy: 81.25%, Loss: 0.4512
    '''
Batch Size: 64, Learning Rate: 0.00025, Tuning Rate: 0.2
{'running_loss': 0.2570601999759674, 'running_binary_acc': 0.8943129777908325, 'loss': 0.22813303768634796, 
 'binary_acc': 0.9034401774406433, 'val_loss': 0.7092078328132629, 'val_binary_acc': 0.753125011920929, 
 'train_steps': 20, 'validation_steps': 3, 'test_loss': 0.507718563079834, 'test_binary_acc': 0.831250011920929}
 
 '''
    
    