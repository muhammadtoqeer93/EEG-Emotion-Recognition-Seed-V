#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python

import numpy as np
import mne
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def get_models(class_weight=None):
    models = dict()
    models['Decision Tree'] = DecisionTreeClassifier() 
    models['K-Nearest Neighbor'] = KNeighborsClassifier()
    models['Gaussian Bayse'] = GaussianNB()
    models['Random Forest']=RandomForestClassifier()
    return models

def run_model_other(model,x_tr,y_tr,x_te,y_te):
    model.fit(x_tr, y_tr)  
    y_pred = model.predict(x_te)   
    print("\n",classification_report(y_te,y_pred))
    print('Accuracy score: ', accuracy_score(y_te, y_pred)*100,"\n")


dataset=pd.read_csv('session1.csv')
y = dataset['label'].values
dataset = dataset.drop('label', axis=1)
x=dataset.values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

models = get_models()
for name, model in models.items():
    print('Name: ',(name))
    run_model_other( model,x_train,y_train,x_test,y_test)


# In[ ]:




