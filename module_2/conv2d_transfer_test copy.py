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
from torchbearer import Trial
import torchbearer
from sklearn.model_selection import train_test_split
from torchbearer import Callback
from torchbearer.callbacks import EarlyStopping
pathFolder = "../module_2/wine_quality/"
os.makedirs(pathFolder,exist_ok=True)
xTrainName = "xTrain1.pkl"
yTrainName = "yTrain.pkl"

with open(pathFolder+xTrainName,'rb') as f1:
    X = pickle.load(f1)

with open(pathFolder+yTrainName,'rb') as f2:
    y = pickle.load(f2)

X_train, X_tv, y_train, y_tv = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tv, y_tv, test_size=0.5, random_state=42)


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imglike_data = torch.tensor(self.data[idx], dtype=torch.float32)

        if self.transform:
            imglike_data = self.transform(imglike_data)
            
        return imglike_data, torch.tensor(self.labels[idx], dtype=torch.float32)


class FCResnet18(nn.Module):
    def __init__(self, tuning_rate, num_hidden_units):
        super(FCResnet18, self).__init__()
        self.trsfRes = models.resnet18(pretrained=True)
        num_ftrs = self.trsfRes.fc.in_features
        self.trsfRes.fc = nn.Identity()  # 기존의 fully connected 레이어를 제거

        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_hidden_units, 1)
        )

        # Freeze the layers according to the tuning_rate
        num_params = len(list(self.trsfRes.parameters()))
        layers_to_freeze = int(num_params * tuning_rate)

        for param in list(self.trsfRes.parameters())[:layers_to_freeze]:
            param.requires_grad = False

    def forward(self, x):
        x = self.trsfRes(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        x = x.squeeze()  # 불필요한 차원 제거
        return x

    

def make_imglike(data, target_size):
    row = math.ceil(target_size/len(data))
    imglike_row = np.tile(data, row)[:target_size]
    imglike = np.tile(imglike_row, (3, target_size, 1))
    return imglike


if __name__ == '__main__':
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    parameters = { 
        'batch_size': [8,16, 32, 64], # 4,8,16, 32
        'lr': [0.00025,0.0001, 0.001, 0.01],  #0.00025가 많이씀   0.01까지
        'tuning_rate': [0.2, 0.4, 0.6,0.8]  # 0.2부터
    }

    best_accuracy = 0.0
    best_parameters = {}
    # EarlyStopping 콜백 생성
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    for batch_size in parameters['batch_size']:
        for lr in parameters['lr']:
            for tuning_rate in parameters['tuning_rate']:
                transform = transforms.Compose([
                    lambda x: make_imglike(x, target_size=32)
                ])

                train_dataset = CustomDataset(X_train, y_train, transform=transform)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                val_dataset = CustomDataset(X_val, y_val, transform=transform)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

                test_dataset = CustomDataset(X_test, y_test, transform=transform)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                # 인스턴스 생성 시 num_hidden_units 인자를 포함하여 제공
                model = FCResnet18(tuning_rate=0.5, num_hidden_units=512).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                # criterion = nn.CrossEntropyLoss()
                criterion = nn.BCELoss()

                trial = Trial(model, optimizer, criterion, metrics=['loss', 'accuracy'],callbacks=[early_stopping]).to(device)
                trial.with_generators(train_generator=train_loader, val_generator=val_loader, test_generator=test_loader)
                history = trial.run(epochs=120)

                result = trial.evaluate(data_key=torchbearer.TEST_DATA)
                #print(result.keys())  # 사용 가능한 모든 키 출력
                test_accuracy = result['test_binary_acc']

                print(f'Batch Size: {batch_size}, Learning Rate: {lr}, Tuning Rate: {tuning_rate}')
                print(history[-1])

                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_history = history[-1]
                    best_parameters = {'batch_size': batch_size, 'lr': lr, 'tuning_rate': tuning_rate}
                    torch.save(model, './conv2d_transfer_best_model.pt')
    
    print("Best Parameters:", best_parameters)
    print("Best Test Accuracy:", best_accuracy)
    print("Best Performance history", best_history)


# Best Test Accuracy: 0.800000011920929
# Best Performance history {'running_loss': 0.2882416546344757, 'running_binary_acc': 0.8837500214576721, 
#                           'loss': 0.2915509343147278, 'binary_acc': 0.8803752660751343, 
#                           'val_loss': 0.555787980556488, 'val_binary_acc': 0.7749999761581421, 
#                           'train_steps': 80, 'validation_steps': 10, 
#                           'test_loss': 0.5686885714530945, 'test_binary_acc': 0.800000011920929}

#'test_loss': 0.5739026665687561, 'test_binary_acc': 0.7749999761581421}
#'test_loss': 0.6069718599319458, 'test_binary_acc': 0.76249998807
#'test_loss': 0.5947000980377197, 'test_binary_acc': 0.78125
#'test_loss': 0.6215370893478394, 'test_binary_acc': 0.737500011920929}