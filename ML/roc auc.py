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

os.chdir(r'C:\adv analytics\Datasets')
boston = pd.read_csv('Boston.csv')


X = boston.drop('medv', axis=1)
y = boston['medv']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022, train_size=0.7)
knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(mean_absolute_error(y_test, y_pred))


# loop
acc = []
Ks = np.arange(1, 16)
for i in Ks:
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc.append(-mean_squared_error(y_test, y_pred))

i_max = np.argmax(acc)
best_k = Ks[i_max]
print('Best n_neigbors=', best_k)
print('Best score=', acc[i_max])
#loop for r2 square 
acc = []
Ks = np.arange(1, 16)
for i in Ks:
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc.append(r2_score(y_test, y_pred))

i_max = np.argmax(acc)
best_k = Ks[i_max]
print('Best n_neigbors=', best_k)
print('Best score=', acc[i_max])


#scaling 
scaler=StandardScaler()
scaler.fit(X_train)
X_trn_scl=scaler.transform(X_train)
X_tst_scl=scaler.transform(X_test)

knn=KNeighborsRegressor(n_neighbors=1)
knn.fit(X_trn_scl,y_train)
y_pred=knn.predict(X_tst_scl)
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test,y_pred))
#loop for r2 score 
acc = []
Ks = np.arange(1, 16)
for i in Ks:
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_trn_scl, y_train)
    y_pred = knn.predict(X_tst_scl)
    acc.append(r2_score(y_test, y_pred))

i_max = np.argmax(acc)
best_k = Ks[i_max]
print('Best n_neigbors=', best_k)
print('Best score=', acc[i_max])

################### concrreate 
concrete=pd.read_csv(r'C:\adv analytics\sir\Cases\Concrete Strength\Concrete_Data.csv')
X=concrete.drop('Strength',axis=1)
y=concrete['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022, train_size=0.7)
scaler=StandardScaler()
scaler.fit(X_train)
X_trn_scl=scaler.transform(X_train)
X_tst_scl=scaler.transform(X_test)

acc = []
Ks = np.arange(1, 16)
for i in Ks:
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_trn_scl, y_train)
    y_pred = knn.predict(X_tst_scl)
    acc.append(r2_score(y_test, y_pred))

i_max = np.argmax(acc)
best_k = Ks[i_max]
print('Best n_neigbors=', best_k)
print('Best score=', acc[i_max])


#---------------grid using pipeline----------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

scaler=StandardScaler()

knn=KNeighborsRegressor()
pipe=Pipeline([('STD',scaler),('KNN',knn)])

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
params={'KNN__n_neighbors':np.arange(1,16)}
knn=KNeighborsRegressor()
gcv=GridSearchCV(pipe,param_grid=params,scoring='r2',cv=kfold)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


###############################---insurance.csv---############################

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.linear_model import LinearRegression
import os
insure=pd.read_csv(r"C:\adv analytics\ML\sir\PML\Cases\Medical Cost Personal\insurance.csv")


#USING LINEARAREGRESSION (R2)
dum_insure=pd.get_dummies(insure,drop_first=True)

X = dum_insure.drop('charges', axis=1)
y = dum_insure['charges']


kfold = KFold(n_splits=5, shuffle=True, random_state=2022)
lr=LinearRegression()
cross_val_score(knn, X, y, cv = kfold)
# ROC AUC
results = cross_val_score(lr, X, y, cv = kfold, scoring='r2')
print(results.mean())

#USING GRID METHOD (R2)
scaler=StandardScaler()

knn=KNeighborsRegressor()
pipe=Pipeline([('STD',scaler),('KNN',knn)])

kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
params={'KNN__n_neighbors':np.arange(1,16)}
knn=KNeighborsRegressor()
gcv=GridSearchCV(pipe,param_grid=params,scoring='r2',cv=kfold)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


##########predicting on unlabelled data 
knn=KNeighborsRegressor(n_neighbors=7)
pipe=Pipeline([('STD',scaler),('KNN',knn)])
pipe.fit(X,y)

tst_insure=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Medical Cost Personal\tst_insure.csv')
dum_tst=pd.get_dummies(tst_insure,drop_first=True)
print(X.dtypes)
print(dum_tst.dtypes)
prediction=pipe.predict(dum_tst)

###########################  OR  ########################
from sklearn.model_selection import GridSearchCV
pd_cv=pd.DataFrame(gcv.cv_results_)
best_model=gcv.best_estimator_
tst_insure=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Medical Cost Personal\tst_insure.csv')
dum_tst=pd.get_dummies(tst_insure,drop_first=True)
prediction=best_model.predict(dum_tst)

#--------------------------------------
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV

image_seg=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Image Segmentation\Image_Segmention.csv')
 
y=image_seg['Class']
X=image_seg.drop['Class']
le=LabelEncoder()
le_y=le.fit_transform(y)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X,le_y,stratify=le_y,
                                                    random_state=2022,train_size=0.7)

scaler=StandardScaler()

knn=KNeighborsClassifier()
pipe=Pipeline([('STD',scaler),('KNN',knn)])

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params={'KNN__n_neighbors':np.arange(1,16)}
knn=KNeighborsRegressor()
gcv=GridSearchCV(pipe,param_grid=params,scoring='neg_log_loss',cv=kfold)

gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)


#-----------------------tst_img-------------------unlabelled data predict class----



import os
txt_img=pd.read_csv(r"C:\adv analytics\ML\sir\PML\Cases\Image Segmentation\tst_img.csv")

best_model=gcv.best_estimator_
prediction=best_model.predict(txt_img)

print(le.inverse_transform(prediction))


####################################### for bank

bank=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Bankruptcy\Bankruptcy.csv')
 
y=bank['D']
X=bank.drop(['NO','D','YR'],axis=1)



scaler=StandardScaler()

knn=KNeighborsClassifier()
pipe=Pipeline([('STD',scaler),('KNN',knn)])

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params={'KNN__n_neighbors':np.arange(1,16)}

gcv=GridSearchCV(pipe,param_grid=params,scoring='roc_auc',cv=kfold)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

#predict on unlabelled data

import os
txt_brupt=pd.read_csv(r"C:\adv analytics\ML\sir\PML\Cases\Bankruptcy\testBankruptcy.csv")
txt_brupt.drop('NO',axis=1,inplace=True)
best_model=gcv.best_estimator_
prediction=best_model.predict(txt_brupt)




