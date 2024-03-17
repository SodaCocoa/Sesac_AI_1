import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,random_split
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import pickle
pathFolder = "./wine_quality/"
xTrainName = "xTrain1.pkl"
yTrainName = "yTrain_onehot.pkl"
with open(pathFolder+xTrainName,'rb') as f1:
    X = pickle.load(f1)
with open(pathFolder+yTrainName,'rb') as f2:
    Y = pickle.load(f2)




# 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# 데이터를 PyTorch 텐서로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# TensorDataset 객체 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# DataLoader 생성
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


#Accuracy: 76.5625%
class WineCNN(nn.Module):
    def __init__(self):
        super(WineCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=11, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.5)#드롭아웃을 추가하여 과적합 방지 실제로 손실이 줄었음
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 2)  # 2개의 출력 클래스 (품질: 좋음/나쁨)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)  # 배치 차원을 제외하고 평탄화
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#연구 노트
#드롭아웃 있을때랑 없을때랑 그래프 비교하기
class betterWineCNN(nn.Module):
    def __init__(self):
        super(WineCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=11, out_channels=64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# 모델 인스턴스 생성
model = betterWineCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# 훈련 세트를 훈련 및 검증 세트로 분할
num_train = len(train_dataset)
num_val = int(0.2 * num_train)
train_ds, val_ds = random_split(train_dataset, [num_train - num_val, num_val])

# DataLoader 업데이트
train_loader = DataLoader(dataset=train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_ds, batch_size=16, shuffle=False)

# 최적의 모델을 저장하기 위한 초기 설정
best_val_loss = float('inf')
best_model = None
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

# 학습 과정
epochs = 50
for epoch in range(epochs):
    train_loss, train_correct = 0, 0
    val_loss, val_correct = 0, 0
    total = 0
    
    # 훈련 부분
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(output, 1)
        train_correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
    
    train_acc = 100 * train_correct / total
    
    # 검증 부분
    model.eval()
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            val_correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    val_acc = 100 * val_correct / total
    
    # 이력 저장
    history['train_loss'].append(train_loss / len(train_loader))
    history['val_loss'].append(val_loss / len(val_loader))
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    # 최적의 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()
        
    print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}, Train Acc: {train_acc}%, '
          f'Val Loss: {val_loss / len(val_loader)}, Val Acc: {val_acc}%')
    
#6단계: 모델 평가
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output = model(X_batch)
        _, predicted = torch.max(output.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f'Accuracy: {100 * correct / total}%')
