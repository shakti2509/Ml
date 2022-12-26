
##################################   GAUSSIAN NB   ####################################
import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import log_loss
brupt=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Bankruptcy\Bankruptcy.csv',index_col=0)

y=brupt['D']
X=brupt.drop(['D','YR'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,random_state=2022, train_size=0.7)

nb=GaussianNB()
nb.fit(X_train,y_train)
y_pred=nb.predict(X_test)
y_pred_prob=nb.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test,y_pred_prob))



####### USING K-FOLD  #######################################

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(nb,X,y,cv=kfold,scoring='roc_auc')
print(results.mean())

###################### for  image segemtation 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

image=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Image Segmentation\Image_Segmention.csv')
 
y=image['Class']
X=image.drop('Class',axis=1)
le= LabelEncoder()
le_y=le.fit_transform(y)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, le_y,stratify=y,random_state=2022, train_size=0.7)

nb=GaussianNB()
nb.fit(X_train,y_train)
y_pred=nb.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=nb.predict_proba(X_test)
print(log_loss(y_test,y_pred_prob))


####### USING K-FOLD  #######################################

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(nb,X,le_y,cv=kfold,scoring='neg_log_loss')
print(results.mean())





##########################  LDA (LINEAR DISTRIMINATE ANALYSIS)   #################################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
 
brupt=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Bankruptcy\Bankruptcy.csv',index_col=0)

y=brupt['D']
X=brupt.drop(['D','YR'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,random_state=2022, train_size=0.7)

de=LinearDiscriminantAnalysis()
de.fit(X_train,y_train)
y_pred=de.predict(X_test)
y_pred_prob=de.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test,y_pred_prob))


####### USING K-FOLD  #######################################

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(de,X,y,cv=kfold,scoring='roc_auc')
print(results.mean())


############ FOR vEHICLE.CSV


vehicle=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Vehicle Silhouettes\Vehicle.csv')
 
y=vehicle['Class']
X=vehicle.drop('Class',axis=1)
le= LabelEncoder()
le_y=le.fit_transform(y)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, le_y,stratify=y,random_state=2022, train_size=0.7)

de=LinearDiscriminantAnalysis()
de.fit(X_train,y_train)
y_pred=de.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=de.predict_proba(X_test)
print(log_loss(y_test,y_pred_prob))

#kfold
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(de,X,y,cv=kfold,scoring='neg_log_loss')
print(results.mean())



##########################  QDA (QUADRATIC DISTRIMINATE ANALYSIS)   #################################
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
 
brupt=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Bankruptcy\Bankruptcy.csv',index_col=0)

y=brupt['D']
X=brupt.drop(['D','YR'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,random_state=2022, train_size=0.7)

qa=QuadraticDiscriminantAnalysis()
qa.fit(X_train,y_train)
y_pred=qa.predict(X_test)
y_pred_prob=qa.predict_proba(X_test)[:,1]
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test,y_pred_prob))


####### USING K-FOLD  #######################################
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(qa,X,y,cv=kfold,scoring='roc_auc')
print(results.mean())

######for vehicle

vehicle=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Vehicle Silhouettes\Vehicle.csv')
 
y=vehicle['Class']
X=vehicle.drop('Class',axis=1)
le= LabelEncoder()
le_y=le.fit_transform(y)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, le_y,stratify=y,random_state=2022, train_size=0.7)

qa=QuadraticDiscriminantAnalysis()
qa.fit(X_train,y_train)
y_pred=qa.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=qa.predict_proba(X_test)
print(log_loss(y_test,y_pred_prob))

###kfold
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(qa,X,y,cv=kfold,scoring='neg_log_loss')
print(results.mean())

###################### SVM ###################
####################  for linear kernel  ##############################


from  sklearn.svm import SVC
from sklearn.model_selection  import GridSearchCV
brupt=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Bankruptcy\Bankruptcy.csv',index_col=0)

y=brupt['D']
X=brupt.drop(['D','YR'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,random_state=2022, train_size=0.7)

svm=SVC(kernel='linear',probability=True,random_state=2022)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob=svm.predict_proba(X_test)
print(log_loss(y_test,y_pred_prob))

####################GRID Search CV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
pipe=Pipeline([('STD',scaler),('SVM',svm)])
              
params={'SVM__C':np.linspace(0.001,10,20)}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='linear',probability=True,random_state=2022)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,verbose=3,scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)




####################  for polynomial kernel  ##############################

from sklearn.model_selection import GridSearchCV
scaler=StandardScaler()
pipe=Pipeline([('STD',scaler),('SVM',svm)])
              

params={'SVM__C':np.linspace(0.001,10,20),'SVM__degree':[2,3,4],'SVM__coef0':np.linspace(-2,4,5)}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='poly',probability=True,random_state=2022)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,verbose=3,scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

################## for radial ############
from sklearn.model_selection import GridSearchCV


scaler=StandardScaler()
pipe=Pipeline([('STD',scaler),('SVM',svm)])

params={'SVM__C':np.linspace(0.001,10,20),'SVM__gamma':np.linspace(0.001,10,20)}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='rbf',probability=True,random_state=2022)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,verbose=3,scoring='roc_auc')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)





#############################  kyphosis.csv  ########################


Kyphosis=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Kyphosis\Kyphosis.csv')
 
y=Kyphosis['Kyphosis']
X=Kyphosis.drop('Kyphosis',axis=1)

le= LabelEncoder()
le_y=le.fit_transform(y)
print(le.classes_)

###  i) for linear:

scaler=StandardScaler()
pipe=Pipeline([('STD',scaler),('SVM',svm)])
              
params={'SVM__C':np.linspace(0.001,10,20)}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='linear',probability=True,random_state=2022)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,verbose=3,scoring='roc_auc')
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)


###  ii)for polynomial:
    
scaler=StandardScaler()
pipe=Pipeline([('STD',scaler),('SVM',svm)])
              

params={'SVM__C':np.linspace(0.001,10,20),'SVM__degree':[2,3,4],'SVM__coef0':np.linspace(-2,4,5)}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='poly',probability=True,random_state=2022)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,verbose=3,scoring='roc_auc')
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)    


###  iii) for radial:
    
scaler=StandardScaler()
pipe=Pipeline([('STD',scaler),('SVM',svm)])

params={'SVM__C':np.linspace(0.001,10,20),'SVM__gamma':np.linspace(0.001,10,20)}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='rbf',probability=True,random_state=2022)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,verbose=3,scoring='roc_auc')
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)




###########################   SVM FOR MORE THAN TWO CLASSES   ###########################

image=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Image Segmentation\Image_Segmention.csv')
 
y=image['Class']
X=image.drop('Class',axis=1)
le= LabelEncoder()
le_y=le.fit_transform(y)
print(le.classes_)

###  i) for linear:
from sklearn.svm import SVC
scaler=StandardScaler()
pipe=Pipeline([('STD',scaler),('SVM',svm)])
              
params={'SVM__C':np.linspace(0.001,10,20),'SVM__decision_function_shape':['ovo','ovr']}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='linear',probability=True,random_state=2022)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,verbose=3,scoring='neg_log_loss')
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)


###  ii)for polynomial:
from sklearn.svm import SVC 
scaler=StandardScaler()
pipe=Pipeline([('STD',scaler),('SVM',svm)])
              

params={'SVM__C':np.linspace(0.001,10,20),'SVM__degree':[2,3,4],'SVM__coef0':np.linspace(-2,4,5),'SVM__decision_function_shape':['ovo','ovr']}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='poly',probability=True,random_state=2022)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,verbose=3,scoring='neg_log_loss')
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)    


###  iii) for radial:
from sklearn.svm import SVC   
scaler=StandardScaler()
pipe=Pipeline([('STD',scaler),('SVM',svm)])

params={'SVM__C':np.linspace(0.001,10,20),'SVM__gamma':np.linspace(0.001,10,20),'SVM__decision_function_shape':['ovo','ovr']}
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
svm=SVC(kernel='rbf',probability=True,random_state=2022)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,verbose=3,scoring='neg_log_loss')
gcv.fit(X,le_y)
print(gcv.best_params_)
print(gcv.best_score_)






#############################  SATELLITE.CSV  ##############################


satellite=pd.read_csv(r'C:\adv analytics\ML\sir\PML\Cases\Satellite Imaging\Satellite.csv', sep=";" )
 
y=satellite['classes']
X=satellite.drop('classes',axis=1)

le= LabelEncoder()
le_y=le.fit_transform(y)
print(le.classes_)

#i)
de=LinearDiscriminantAnalysis()
de.fit(X,le_y)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(de,X,y,cv=kfold,scoring='neg_log_loss')
print(results.mean())


#ii)

qe=QuadraticDiscriminantAnalysis()
qe.fit(X,le_y)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(qe,X,y,cv=kfold,scoring='neg_log_loss')
print(results.mean())

#iii)
nb=GaussianNB()
nb.fit(X,le_y)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
results=cross_val_score(nb,X,y,cv=kfold,scoring='neg_log_loss')
print(results.mean())
