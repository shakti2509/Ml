import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,cross_val_score

mower=pd.read_csv(r'C:\adv analytics\Datasets\RidingMowers.csv')
dum_mow=pd.get_dummies(mower,drop_first=True)

X=dum_mow.drop('Response_Not Bought',axis=1)
y=dum_mow['Response_Not Bought']
knn=KNeighborsClassifier(n_neighbors=3)
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

#Acuracy
cross_val_score(knn,X,y, cv=kfold)
#ROC  AUC
results=cross_val_score(knn,X,y,cv=kfold,scoring='roc_auc')
print(results.mean())

#for loop roc auc

acc=[]
Ks=[1,3,5,7,9,11,13,15]
for i in Ks:
    knn=KNeighborsClassifier(n_neighbors=i)
    results=cross_val_score(knn,X,y,cv=kfold,scoring='roc_auc')
    acc.append(results.mean())
    
i_max=np.argmax(acc)
best_k=Ks[i_max]
print('Best n_neigbors=',best_k)
print("Best Score =",acc[i_max]) 

#########Grid Search cv loop run inside and give a answer 
from sklearn.model_selection import GridSearchCV
params={ 'n_neighbors': [1,3,5,7,9,11,13,15] }
knn=KNeighborsClassifier()
gcv=GridSearchCV(knn, param_grid=params,scoring='roc_auc',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


############## for Bsoton
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
 
boston=pd.read_csv(r'C:\adv analytics\Datasets\Boston.csv')
X=boston.drop('medv',axis=1)
y=boston['medv']
kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
params={ 'n_neighbors': np.arange(1,16) }
knn=KNeighborsRegressor()
gcv=GridSearchCV(knn, param_grid=params,scoring='r2',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

