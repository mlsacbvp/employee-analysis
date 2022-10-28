#!/usr/bin/env python
# coding: utf-8

# In[1]:


# MLPClassifier

import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

X_train = pd.read_csv('emp_x_train.csv')
X_test = pd.read_csv('emp_x_test.csv')
y_train = pd.read_csv('emp_y_train.csv')['Attrition']
y_test = pd.read_csv('emp_y_test.csv')['Attrition']
mlp = MLPClassifier()
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, y_pred)
print(mlp_accuracy)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

#Using GridSearchCV for Hyperparamter Tuning

parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels

clf.best_params_

y_pred2 = clf.predict(X_test)
mlp_accuracy = accuracy_score(y_test, y_pred2)
print(mlp_accuracy)

y_pred2

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred2))

#Although the accuracy after using GridSearchCV is good, it seems the model is predicting everything as 0 simply because most entries in our training set are 0s. This has led to overfitting.

