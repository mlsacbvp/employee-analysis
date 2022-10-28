#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

X_train = pd.read_csv('emp_x_train.csv')
X_test = pd.read_csv('emp_x_test.csv')
y_train = pd.read_csv('emp_y_train.csv')['Attrition']
y_test = pd.read_csv('emp_y_test.csv')['Attrition']

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
gnb_accuracy = accuracy_score(y_test, y_pred)
print(gnb_accuracy)

#Using GridSearchCV for Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
pipe = Pipeline(steps=[
                    ('pca', PCA()),
                    ('estimator', GaussianNB()),
                    ])
    
parameters = {'estimator__var_smoothing': [1e-11, 1e-10, 1e-9]}
Bayes = GridSearchCV(pipe, parameters, scoring='accuracy', cv=10).fit(X_train, y_train)
y_pred2 = Bayes.best_estimator_.predict(X_test)

gnb_accuracy = accuracy_score(y_pred2, y_test) 
print(gnb_accuracy)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred2))

