import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


job=pd.read_csv(r'C:\adv analytics\Datasets\JobSalary2.csv')

##Finding NAs in columns 
job.isnull().sum()

#droping the rows with NA values 
job.dropna()

#constant Imputer
from sklearn.impute import  SimpleImputer
imp=SimpleImputer(strategy='constant',fill_value=50)
imp.fit_transform(job)


job.mean()
#mean Imputer
imp=SimpleImputer(strategy='mean')
imp.fit_transform(job)

#median Imputer
imp=SimpleImputer(strategy='median')
np_imp=imp.fit_transform(job)


pd_imp=pd.DataFrame(np_imp,columns=job.columns)



##################  chemical process  ##################################\
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

chemdata=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Chemical Process Data\ChemicalProcess.csv')


chemdata.isnull().sum()

X=chemdata.drop('Yield',axis=1)
y=chemdata['Yield']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=2022,train_size=0.7)


#mean Imputer
chemdata.mean()
imp=SimpleImputer(strategy='mean') 
# for mean -1.290665707638071 and median is -0.43035543790087405
X_trn_tf=imp.fit_transform(X_train)

X_tst_tf=imp.transform(X_test)

lr = LinearRegression()
lr.fit(X_trn_tf,y_train)

y_pred = lr.predict(X_tst_tf)
print(r2_score(y_test, y_pred))
######## with Pipline 
from sklearn.pipeline import Pipeline
imp=SimpleImputer(strategy='median')
lr=LinearRegression() 
pipe=Pipeline([('IMPUTE',imp),('LR',lr)])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
print(r2_score(y_test,y_pred)


############# K-NN #############

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,GridSearchCV

imp=SimpleImputer(strategy='mean')
scalar=StandardScaler() 
knn=KNeighborsRegressor()
pipe=Pipeline([('IMPUTE',imp),('STD',scalar),('KNN',knn)])     
      

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
params={'IMPUTE__strategy':['mean','median'],'KNN__n_neighbors':np.arange(1,11)}
gcv=GridSearchCV(pipe,param_grid=params,scoring='r2',cv=kfold)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
     
      
boston=pd.read_csv()
      

    