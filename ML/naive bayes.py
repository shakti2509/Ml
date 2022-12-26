import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score


hr = pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\human-resources-analytics\HR_comma_sep.csv')
dum_hr=pd.get_dummies(hr,drop_first=True)

X=dum_hr.drop('left',axis=1)
y=dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022, train_size=0.7)

lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
y_pred_prob=lr.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test,y_pred_prob))


################## Grid search CV#################
from sklearn.model_selection import GridSearchCV,StratifiedKFold
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
lr=LogisticRegression(solver='saga',random_state=2022)
params={'penalty':['l1','l2','elasticnet',None],
                     'C':np.linspace(0.001,4,5),'l1_ratio':np.linspace(0.001,1,5)}
gcv=GridSearchCV(lr, param_grid=params,verbose=3,scoring='roc_auc',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

# by scaling 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
scaler=StandardScaler()
lr=LogisticRegression(solver='saga',random_state=2022)
pipe=Pipeline([('STD',scaler),('LR',lr)])
params={'LR__penalty':['l1','l2','elasticnet',None],
                     'LR__C':np.linspace(0.001,4,5),'LR__l1_ratio':np.linspace(0.001,1,5)}
gcv=GridSearchCV(pipe, param_grid=params,verbose=3,scoring='roc_auc',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


#################### for bank
bank=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Bankruptcy\Bankruptcy.csv')
 
y=bank['D']
X=bank.drop(['NO','D','YR'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=2022, train_size=0.7)

lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
y_pred_prob=lr.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test,y_pred_prob))

# by gridsearch without scaling
from sklearn.model_selection import GridSearchCV,StratifiedKFold
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
lr=LogisticRegression(solver='saga',random_state=2022)
params={'penalty':['l1','l2','elasticnet',None],
                     'C':np.linspace(0.001,4,5),'l1_ratio':np.linspace(0.001,1,5)}
gcv=GridSearchCV(lr, param_grid=params,verbose=3,scoring='roc_auc',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)



# by scaling 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
scaler=StandardScaler()
lr=LogisticRegression(solver='saga',random_state=2022)
pipe=Pipeline([('STD',scaler),('LR',lr)])
params={'LR__penalty':['l1','l2','elasticnet',None],
                     'LR__C':np.linspace(0.001,4,5),'LR__l1_ratio':np.linspace(0.001,1,5)}
gcv=GridSearchCV(pipe, param_grid=params,verbose=3,scoring='roc_auc',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)



###############image segmentation  for multi classes
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
image_seg=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Image Segmentation\Image_Segmention.csv')
 
y=image_seg['Class']
X=image_seg.drop('Class',axis=1)
le= LabelEncoder()
le_y=le.fit_transform(y)
print(le.classes_)

scaler=StandardScaler()
lr=LogisticRegression(solver='saga',random_state=2022)
pipe=Pipeline([('STD',scaler),('LR',lr)])
params={'LR__penalty':['l1','l2','elasticnet',None],
                     'LR__C':np.linspace(0.001,4,5),'LR__l1_ratio':np.linspace(0.001,1,5),'LR__multi_class':['ovr','multinomial']}
gcv=GridSearchCV(pipe, param_grid=params,verbose=3,scoring='neg_log_loss',cv=kfold)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


########################################   vehicle silhpuetles ######################


vehicle=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Vehicle Silhouettes\Vehicle.csv')
 
y=vehicle['Class']
X=vehicle.drop(['Class'],axis=1)

le= LabelEncoder()
le_y=le.fit_transform(y)
print(le.classes_)

scaler=StandardScaler()
lr=LogisticRegression(solver='saga',random_state=2022)
pipe=Pipeline([('STD',scaler),('LR',lr)])
params={'LR__penalty':['l1','l2','elasticnet',None],
                     'LR__C':np.linspace(0.001,4,5),'LR__l1_ratio':np.linspace(0.001,1,5),'LR__multi_class':['ovr','multinomial']}
gcv=GridSearchCV(pipe, param_grid=params,verbose=3,scoring='neg_log_loss',cv=kfold)
gcv.fit(X,y)
###############################################################################
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

telecom=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Telecom\Telecom.csv')

dum_tel=pd.get_dummies(telecom,drop_first=True)
 
y=dum_tel['Response_Y']
X=dum_tel.drop(['Response_Y'],axis=1)


nb= BernoulliNB()
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,random_state=2022, train_size=0.7)

nb=BernoulliNB()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
y_pred_prob=lr.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test,y_pred_prob))


#########################################################################
from sklearn.model_selection import StratifiedKFold,cross_val_score


telecom=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Telecom\Telecom.csv')

dum_tel=pd.get_dummies(telecom,drop_first=True)
 
y=dum_tel['Response_Y']
X=dum_tel.drop(['Response_Y'],axis=1)


nb=BernoulliNB()
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(nb,X,y,cv=kfold,scoring='roc_auc')
print(results.mean())

################################## Cancer.csv ######################################

telecom=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Cancer\Cancer.csv')

dum_tel=pd.get_dummies(telecom,drop_first=True)
 
y=dum_tel['Class_recurrence-events']
X=dum_tel.drop(['Class_recurrence-events','subjid'],axis=1)


nb= BernoulliNB()
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,random_state=2022, train_size=0.7)

nb=BernoulliNB()
nb.fit(X_train,y_train)
y_pred=lr.predict(X_test)
y_pred_prob=nb.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test,y_pred_prob))


kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(nb,X,y,cv=kfold,scoring='roc_auc')
print(results.mean())

