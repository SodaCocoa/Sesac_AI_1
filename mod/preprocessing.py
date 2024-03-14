import pandas as pd
import numpy as np
import os
import pickle

csv_folder_path = "./csv_data/"
total_data_list = os.listdir(csv_folder_path)


##### 변수 입력
csv_name = 'apple_quality.csv'
stretch = False
#####


csv_path = csv_folder_path+csv_name

trainDf = pd.read_csv(csv_path)
trainDf.dropna(inplace=True)

# object를 float으로 변환시도, 에러가 뜨는지.
for idx,aci in enumerate(trainDf['Acidity']):
    try:
        float(aci)
    except:
        print(idx,aci)

# object data를 float으로 변환
trainDf['Acidity'] = trainDf['Acidity'].astype(float)

# 쓸모없는 정보 drop
trainDf = trainDf.drop(['A_id'],axis=1)

yTrain = trainDf['Quality']
xTrain = trainDf.drop(['Quality'],axis=1)

yTrain = pd.get_dummies(yTrain, columns=['Quality'],dtype=int)

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# xTrain의 모든 컬럼에 최소-최대 정규화 적용
xTrain = xTrain.apply(min_max_normalize)
xTrain_stretch = xTrain.apply(lambda x:(2*x-1))

pathFolder = "./train/apple/"
os.makedirs(pathFolder,exist_ok=True)

xTrainName = "xTrain.pkl"
yTrainName = "yTrain.pkl"

xTrainNp = xTrain.values
yTrainNp = yTrain.values

xTrainName_s = "xTrain_stretch.pkl"
yTrainName_s = "yTrain_stretch.pkl"

xTrainNp_s = xTrain_stretch.values
yTrainNp_s = yTrain.values

print(xTrainNp[0])
print(xTrainNp_s[0])
def save_to_pkl(xTrainName, xTrainNp,yTrainName,yTrainNp):
    with open(pathFolder+xTrainName,'wb') as f1:
        pickle.dump(xTrainNp,f1)
    with open(pathFolder+yTrainName,'wb') as f2:
        pickle.dump(yTrainNp,f2)

#save_to_pkl(xTrainName,xTrainNp,yTrainName,yTrainNp)
#save_to_pkl(xTrainName_s,xTrainNp_s,yTrainName_s,yTrainNp_s)