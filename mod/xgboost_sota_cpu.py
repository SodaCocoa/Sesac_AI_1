import numpy as np
import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import random
import time
import re

class for_any_folder():
    def __init__(self,go_once,choose_data='') -> None:
        self.base_folder = "./train/"
        os.makedirs(self.base_folder,exist_ok=True)
        self.list_train = os.listdir(self.base_folder)
        # 시작하는 하이퍼파라미터 검색 공간 정의
        self.go_once = go_once
        self.choose_data = choose_data
        
    def data_preprocessing(self):        
        self.path_folder = self.base_folder+self.name_folder
        xTrainName = "/xTrain1.pkl"
        yTrainName = "/yTrain_onehot.pkl"
        with open(self.path_folder+xTrainName,'rb') as f1:
            X = pickle.load(f1)

        with open(self.path_folder+yTrainName,'rb') as f2:
            y = pickle.load(f2)

        # 데이터를 훈련 세트와 테스트 세트로 분할
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
    def train_with_folder(self):
        result_path = f'./result/{self.name_folder}/'
        os.makedirs(result_path,exist_ok=True)
        fgm = find_global_max(self.X_train,self.y_train,self.X_test,self.y_test,result_path,go_once=self.go_once)
        sota = fgm.find_sota()
        accuracy = sota[0]
        hyperparameter = sota[1]
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        os.makedirs(result_path,exist_ok=True)
        path_text = result_path+'/check.txt'
        if os.path.exists(path_text):
            with open(path_text, 'r') as file:
                first_line = file.readline().strip()  # strip() is used to remove the newline character at the end
                # Extract all numbers from the first line
                numbers = re.findall(r'\d+\.?\d*', first_line)  # This regex will find all integers and floating point numbers
                numbers = float(numbers[0])
        if accuracy > numbers:
            with open(path_text,'w') as file:
                file.write(f"accuracy : {accuracy}\n")
                file.write(str(hyperparameter)+"\n")
                file.write("\n")
        
            # Save 'accuracy' to 'best_acc.pkl'
            accuracy_file_path = f"{result_path}/best_acc.pkl"
            with open(accuracy_file_path, 'wb') as f:
                pickle.dump(accuracy, f)

            # Save 'hyperparameters' to 'best_hyper.pkl'
            hyperparameters_file_path = f"{result_path}/best_hyper.pkl"
            with open(hyperparameters_file_path, 'wb') as f:
                pickle.dump(hyperparameter, f)
            
            print("best accuracy is updated")
    
    def total_processing(self):
        if self.choose_data != '':
            if type(self.choose_data) == list:
                self.list_train = self.choose_data
            else:
                self.list_train = [self.choose_data]
        
        for i in self.list_train:
            self.name_folder = i
            self.data_preprocessing()
            self.train_with_folder()

#data_train = xgb.DMatrix(X_train,y_train)
#data_test = xgb.DMatrix(X_test,y_test)

# 하이퍼파라미터 검색 공간 정의
"""param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.5, 1],
    'reg_alpha': [0, 1, 5],
    'reg_lambda': [1, 5, 10],
}

"""

class find_global_max():
    def __init__(self,X_train,y_train,X_test,y_test,path_folder,start_param={},
                 mm_param={},diff_param={},go_once=False) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.path_folder = path_folder
        
        if start_param == {}:
            start_param = {
                'max_depth':  8,
                'learning_rate':  0.4,
                'n_estimators': 50,
                'subsample': 0.3+0.7*random.random(),
                'colsample_bytree': 0.3+0.7*random.random(),
                'gamma': 0,
                'reg_alpha': 0.1,
                'reg_lambda': 4     
            }
        if go_once == True:
            start_param = {
            'max_depth':  8,
            'learning_rate':  0.4,
            'n_estimators': 50,
            'subsample': 1,
            'colsample_bytree': 1,
            'gamma': 0,
            'reg_alpha': 0.2,
            'reg_lambda': 5
            }

        if mm_param == {}:
            mm_param = {
                'max_depth': [ 2,11],
                'learning_rate': [ 0.1,0.9],
                'n_estimators': [ 50,50],
                'subsample': [0,1.0],
                'colsample_bytree': [0,1.0],
                'gamma': [0,1],
                'reg_alpha': [0,1],
                'reg_lambda': [0,11]
            }
        if diff_param =={}:
            diff_param = {
                'max_depth': 1,
                'learning_rate': [0.1,0.7],
                'n_estimators': 50,
                'subsample': [0.1,0.7],
                'colsample_bytree': [0.1,0.7],
                'gamma': [0.1,0.7],
                'reg_alpha': [0.1,0.7],
                'reg_lambda': [1,0.6]
            }
        self.start_param = start_param
        self.mm_param = mm_param
        self.diff_param = diff_param
        self.current_param = self.start_param.copy()
        self.accuracy = 0
        self.best_param = self.start_param.copy()

        self.go_once = go_once
        self.patients = -2
        self.go_deeper = False
        self.no_best = True
        self.ratio = 1

    def find_sota(self):        
        while self.patients <6:
            if self.patients > 3:
                self.go_deeper = True
            if self.go_deeper == True:
                self.ratio += 1
            self.set_search_area()
            self.patients += 1
            
            self.one_step()
            if self.go_once==True:
                break
        return self.accuracy,self.best_param

    def set_search_area(self):
        new_dict = dict()
        print(self.current_param)
        for i in self.best_param:
            area = []
            area.append(self.best_param[i])
            if self.go_once != True:
                mm0 = self.mm_param[i]
                diff = 0
                if type(self.diff_param[i])==list:
                    if self.go_deeper == False:
                        diff = self.diff_param[i][0]
                    else:
                        diff = self.diff_param[i][0]*self.diff_param[i][1]*self.ratio
                else:
                    diff = self.diff_param[i]
                if self.no_best==True:
                    diff = 2*diff
                
                if self.best_param[i]-diff>=mm0[0]:                
                    area.append(round(self.best_param[i]-diff,6))
                else:
                    area.append(mm0[0])
                if self.best_param[i]+diff<=mm0[1]:
                    area.append(round(self.best_param[i]+diff,6))
                else:
                    area.append(mm0[1])
            area = list(set(area))
            area.sort()
            new_dict[i] = area
        self.current_param = new_dict
        self.go_deeper = False
        

    def one_step(self):
        start_time = time.time()
        # XGBoost 분류기 인스턴스 생성
        xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        
        # 그리드 탐색 객체 초기화
        grid_search = GridSearchCV(xgb_clf, self.current_param, cv=3, scoring='accuracy', verbose=0)
       
        # 그리드 탐색 수행
        grid_search.fit(self.X_train,self.y_train)

        # 최적 모델로 예측
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(self.X_test)

        # 예측 정확도 평가
        accuracy = accuracy_score(self.y_test, predictions)
        best_model.save_model(self.path_folder +'best_model.json')
        if accuracy > self.accuracy:
            print("========================")
            print("data : "+self.path_folder.split('/')[-2])
            print("time elapsed : "+str(round(time.time()-start_time))+" s | self.patients : "+str(self.patients))
            print("accuracy is updated :",accuracy)
            hyperparameter_text = "Local best parameters found: "+ str(grid_search.best_params_)
            print(hyperparameter_text)

            self.accuracy = accuracy 
            self.best_param = grid_search.best_params_
            self.patients = 0
            self.go_deeper = True
            # 모델을 JSON 형식으로 저장
            
### 변수 입력
go_once = True
###
if __name__ =="__main__":
    print("가지고 있는 데이터들 :",os.listdir('./train'))
    a = input("적용할 데이터를 입력해 주세요. 전체 = all  :  ")
    b = input("적당한 파라미터에 대해서 한번만 할까요? 1:yes, 0:no  :  ")
    a = '' if a=='all' else a
    b = True if b=="1" else False
    faf = for_any_folder(go_once=b,choose_data=a)
    faf.total_processing()

