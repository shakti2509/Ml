from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import  pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

boston=pd.read_csv(r'C:\adv analytics\sir\Datasets\Boston.csv')

X=boston.drop('medv',axis=1)
y=boston['medv']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=2022,train_size=0.7)

ridge=Ridge(alpha=2.5)
ridge.fit(X_train,y_train)
y_pred=ridge.predict(X_test)
print(r2_score(y_test,y_pred))


#######grind search CV
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
ridge=Ridge()
params={'alpha':np.linspace(0.001,11,20)}
gcv=GridSearchCV(ridge,param_grid=params,cv=kfold,scoring='r2')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model=gcv.best_estimator_
print(best_model.coef_)

#############   concrete.csv   ############################


con=pd.read_csv(r'C:\adv analytics\sir\Cases\Concrete Strength\Concrete_Data.csv')
X=con.drop('Strength',axis=1)
y=con['Strength']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=2022,train_size=0.7)

ridge=Ridge(alpha=2.5)
ridge.fit(X_train,y_train)
y_pred=ridge.predict(X_test)
print(r2_score(y_test,y_pred))


#######grind search CV
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
ridge=Ridge()
params={'alpha':np.linspace(0.001,11,20)}
gcv=GridSearchCV(ridge,param_grid=params,cv=kfold,scoring='r2')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model=gcv.best_estimator_
print(best_model.coef_)


############### Lasoo
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import  pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split


con=pd.read_csv(r'C:\adv analytics\sir\Cases\Concrete Strength\Concrete_Data.csv')
X=con.drop('Strength',axis=1)
y=con['Strength']



#######grind search CV
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
lasso=Lasso()
params={'alpha':np.linspace(0.001,11,20)}
gcv=GridSearchCV(lasso,param_grid=params,cv=kfold,scoring='r2')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model=gcv.best_estimator_
print(best_model.coef_)

#elastic net 
from sklearn.linear_model import ElasticNet
kfold=KFold(n_splits=5,shuffle=True,random_state=2022)
elastic=ElasticNet()
params={'alpha':np.linspace(0.001,11,20),'l1_ratio':np.linspace(0,1,5)}
gcv=GridSearchCV(elastic,param_grid=params,cv=kfold,scoring='r2')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model=gcv.best_estimator_
print(best_model.coef_)













