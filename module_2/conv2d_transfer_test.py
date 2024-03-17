import numpy as np
import os
import math
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split

# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imglike_data = self.data[idx].astype(np.float32)
        if self.transform:
            imglike_data = self.transform(imglike_data)
        label = self.labels[idx]
        return imglike_data, label

# 이미지 형태로 데이터 변환
def make_imglike(data, target_size=224):
    row = math.ceil(target_size / data.shape[0])
    imglike_row = np.tile(data, (row, 1))
    imglike = np.tile(imglike_row, (3, 1, 1))[:target_size, :target_size, :]
    return imglike

# 모델 정의
class TransferResnet18(nn.Module):
    def __init__(self, num_classes):
        super(TransferResnet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# 메인 코드
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pathFolder = "../module_2/wine_quality/"
    xTrainName = "xTrain1.pkl"
    yTrainName = "yTrain_onehot.pkl"

    with open(os.path.join(pathFolder, xTrainName), 'rb') as f1:
        X = pickle.load(f1)

    with open(os.path.join(pathFolder, yTrainName), 'rb') as f2:
        y = pickle.load(f2)  # 원-핫 인코딩된 라벨을 정수 라벨로 변환해야 할 수 있습니다.

    X_train, X_tv, y_train, y_tv = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tv, y_tv, test_size=0.5, random_state=42)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(X_train, y_train, transform=lambda x: transform(make_imglike(x)))
    val_dataset = CustomDataset(X_val, y_val, transform=lambda x: transform(make_imglike(x)))
    test_dataset = CustomDataset(X_test, y_test, transform=lambda x: transform(make_imglike(x)))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_classes = len(np.unique(y_train))  # 클래스 수 조정 필요
    model = TransferResnet18(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 학습 과정
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    # 평가 과정
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy on test set: {100 * correct / total}%")
