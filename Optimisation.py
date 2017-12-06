# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:32:07 2017

@author: Quentin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


Xtrain=pd.read_table("X_train.txt",sep='\s+',header=None)
ytrain=pd.read_table("y_train.txt",sep='\s+',header=None,names=('y'))
Xtest=pd.read_table("X_test.txt",sep='\s+',header=None)
ytest=pd.read_table("y_test.txt",sep='\s+',header=None,names=('y'))
ytrain=ytrain["y"]
ytest=ytest["y"]
LABELS = ["WALKING","WALKING UPSTAIRS","WALKING DOWNSTAIRS","SITTING","STANDING","LAYING"]

## Methodes

method = LinearSVC
if method is LinearSVC:
    param=[{"C":[0.9,1,1.1]}] #0.1,0.5,0.75,1.5,2,3,10
        #penalty='l2',
        #               loss='squared_hinge',
         #              dual=True,
           #            tol=0.0001,
            #           C=c,
             #          multi_class='ovr',
              #         fit_intercept=True,
               #        intercept_scaling=1,
                #       class_weight=None




ts = time.time()
model = GridSearchCV(method(), param,cv=10,n_jobs=4)
Resopt=model.fit(Xtrain, ytrain)
te = time.time()

print("Results:",Resopt.cv_results_)
print("Meilleur Score = %f, Meilleur paramètre = %s" % (Resopt.best_score_,Resopt.best_params_))
prev=Resopt.predict(Xtest)
print("Test Score=",accuracy_score(prev,ytest))
print("Time running : %d secondes" %(te-ts))