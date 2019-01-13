# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 22:00:36 2018

@author: archit bansal
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test=test.drop(["id"],axis=1)
X_train = train.drop(["price_range"],axis=1)
y_train = train["price_range"]

#analyzing the data
'''


'''
train.info()
train.describe()

#data visualization

#buliding the optimal data using automatic backward elimnation
import statsmodels.formula.api as sm
SL = 0.05
X_train_arr=X_train.values
X_opt = X_train_arr[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
X=np.append(arr=np.ones((2000,1)).astype(int),values=X_train,axis=1)
regressor_ols=sm.OLS(endog=y_train,exog=X_opt).fit()
regressor_ols.summary()
li=[]
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y_train, x).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j] == maxVar):
                    x = np.delete(x, j, 1)
                    li.append(j)
    regressor_OLS.summary()
    return x
X_Modeled = backwardElimination(X_opt, SL)
test=test.values
test=np.delete(test,5,1)
test=np.delete(test,4,1)
test=np.delete(test,13,1)


#applying feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Modeled= sc.fit_transform(X_Modeled)
test = sc.fit_transform(test)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_Modeled, y_train, test_size = 0.3, random_state = 0)


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train1, y_train1)
# Predicting the Test set results
y_pred_SVC= classifier.predict(X_test1)
#calculating accuracy
acc_SVC= round(classifier.score(X_train1,y_train1) * 100, 2)
print(acc_SVC)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_SVC= confusion_matrix(y_test1, y_pred_SVC)


#fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train1,y_train1)
# Predicting the Test set results
y_pred_logistic= classifier.predict(X_test1)
#calculating accuracy
acc_logistic= round(classifier.score(X_train1,y_train1) * 100, 2)
print(acc_SVC)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_logistic= confusion_matrix(y_test1, y_pred_logistic)


#fitting the knn_calssifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train1,y_train1)
# Predicting the Test set results
y_pred_knn= classifier.predict(X_test1)
#calculating accuracy
acc_knn= round(classifier.score(X_train1,y_train1) * 100, 2)
print(acc_knn)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn= confusion_matrix(y_test1, y_pred_knn)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train1,y_train1)
# Predicting the Test set results
y_pred_naive= classifier.predict(X_test1)
#calculating accuracy
acc_naive= round(classifier.score(X_train1,y_train1) * 100, 2)
print(acc_naive)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_naive= confusion_matrix(y_test1, y_pred_naive)

# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
classifier.fit(X_train1,y_train1)
# Predicting the Test set results
y_pred_random= classifier.predict(X_test1)
#calculating accuracy
acc_random= round(classifier.score(X_train1,y_train1) * 100, 2)
print(acc_random)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_random= confusion_matrix(y_test1, y_pred_random)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test1,y_pred_random))

#prdicting results for test set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
classifier.fit(X_Modeled,y_train)
predicted=classifier.predict(test)
















