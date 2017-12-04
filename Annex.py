# -*- coding: utf-8 -*-
"""
Created on Dec 2 10:53:27 2017

MACHINE LEARNING OPTIMIZER

@author: Quentin


"""

import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import sys
import copy


Methods_names={LinearSVC:"LinearSVC",
         SVC:"SVC",
         LogisticRegression:"LogisticRegression",
         XGBClassifier:"XGBClassifier"         
         }
Methods={"LinearSVC":LinearSVC,
         "SVC":SVC,
         "LogisticRegression":LogisticRegression,
         "XGBClassifier":XGBClassifier         
         }
         

class result:
    def __init__(self, method, params, cv_score):
        self.method=method
        self.cv_score=cv_score
        self.params=params

    def __str__(self):
        print('%s:\nParameters:'%Methods_names[self.method])
        print(self.params)
        print('Score: %0.3e' %self.cv_score)
        return ''
        

class Results:
    def __init__(self,list=[]):
        self.nb_results=len(list)
        self.list_results=list
        self.total_time=0
        self.wrong_configs=[]
        
    def add_result(self,result):
        self.list_results.append(result)
        self.nb_results+=1
        
    def __str__(self):
        print('%d Results:' %self.nb_results)
        for r in self.list_results:
            print(r)
        return '\n'

    def sort(self,reverse=False):
        return Results(sorted(self.list_results, key=operator.attrgetter('cv_score'), reverse=reverse))

    def best_score(self,max=True):
        return sorted(self.list_results, key=operator.attrgetter('cv_score'), reverse=max)[0].cv_score

    def bests(self,max=True,nb=1):
        return sorted(self.list_results, key=operator.attrgetter('cv_score'), reverse=max)[:nb]
        
    def isnan(self,obj):
        if type(obj) is Results:
            l=[]
            for r in obj.list_results:
                l.append(np.isnan(r.cv_score))

        return l
        
    def say_hello(self):
        if self.nb_results==0:
            print('Welcome, I am Sammy Nimizz.\nMay I help you?')
        else:
            print('Welcome back, I am Sammy Nimizz.\nI have already run %d models and got a best score of %0.3e.\n May I help you? (I am %d secondes old)'%(self.nb_results,self.best_score(),self.total_time))

            
    def run(self,Xtrain,ytrain,method,param,time_limit=3600):
        start=time.time()
        for parm in ParameterGrid(param):
            if self.check_isnew(method,parm):
                ts = time.time()
                scores=cross_val_score(method(**parm), Xtrain, ytrain, cv=10)
                te = time.time()
                self.total_time+=(te-ts)
                self.add_result(result(method,parm,scores.mean()))
                if (time.time()-start)>(time_limit):
                    print('Your time limit was reached.\n See you soon!')
                    break
            else:
                print('This configuration was already ran.',parm)
            
    def check_isnew(self,method,param):
        res=True
        for r in self.list_results:
            if method==r.method:
                if param==r.params:
                    res= False
                    break
        for p in self.wrong_configs:
            if p==param:
                res=False
                break
        return res
                  
                                
    def add_wrong_config(self,param):
        self.wrong_configs.append(param)
        return 0

        
    def random_start(self,Xtrain,ytrain,nb_run=10,methods='all'):
        params={}
            
        for run in range(nb_run):
            if methods is not 'all':
                if type(methods) is not list:
                    method=methods
                else:
                    method=methods[np.random.randint(0,len(methods)-1)]
                if method not in params.keys():
                    params.update({method:get_param_choices(method)})
                ind=np.random.randint(0,len(ParameterGrid(params[method]))-1)
                new_param=ParameterGrid(params[method])[ind]
            else:
                ind_method=np.random.randint(0,len(Methods)-1)
                method=Methods[list(Methods.keys())[ind_method]]
                print(method)
                if method not in params.keys():
                    params.update({method:get_param_choices(method)})
                ind=np.random.randint(0,len(ParameterGrid(params[method]))-1)
                new_param=ParameterGrid(params[method])[ind]
            if self.check_isnew(method,new_param):
                self.run_cross_val(Xtrain,ytrain,method,new_param)
    
    
    def optimize(self,Xtrain,ytrain,nb_epoch=5,method='all',goal='max',coef_selection=0.2):
        print('I want to improve myself!')
        if self.nb_results<10:
            print("I am too young to be optimized, let me try some randomness first.")
            self.random_start(Xtrain,ytrain,nb_run=10,methods=method)
        for epoch in range(nb_epoch):
            print('\nEpoch',epoch+1)
            pop=self.select(coef_selection,method,goal)
            pop=pop.breed()
            pop=pop.mutate()
            self.update(pop,Xtrain,ytrain)
        
    def update(self,pop, Xtrain,ytrain):
        print('Evaluation...')
        for r in pop.list_results:
            if self.check_isnew(r.method,r.param):
                self.run_cross_val(Xtrain,ytrain,r.method,r.params)
    
    def run_cross_val(self,Xtrain,ytrain,method,param):
        print('Running %s with parameters: %s'%(Methods_names[method],param))            
        ts = time.time()

        try:
            scores=cross_val_score(method(**param), Xtrain, ytrain, cv=10)
            score=scores.mean()
            print('Score: %0.3e\n'%score)
            self.add_result(result(method,param,score))
        
        except TypeError as err:
            print("Type error: {0}".format(err))
        except:
            print("An error occured with this configuration.", sys.exc_info()[0])
            self.add_wrong_config(param)
        te = time.time()
        self.total_time+=(te-ts)
    
    
    def breed(self):
        return self
        
    def mutate(self):
        print('Mutation!')
        params={}
        for r in self.list_results:
            if r.method not in params.keys():
                params.update({r.method:get_param_choices(r.method)})
            try:
                ind=list(ParameterGrid(params[r.method])).index(r.param)
                new_ind=np.random.poisson(ind)
                new_param=ParameterGrid(params[r.method])[new_ind]
                
            except:
                new_ind=np.random.randint(0,len(ParameterGrid(params[r.method]))-1)
                new_param=ParameterGrid(params[r.method])[new_ind]
            r.param=new_param
            r.cv_score=0
        return self
    
    def select(self,coef_selection,method,goal):
        nb_selected=round(self.nb_results*coef_selection)
        if goal=='max':
            bests=self.sort(True)
        else:
            bests=self.sort(False)
        if method!='all':
            bests=bests.restrict_to_methods(method)
            nb_selected=min(bests.nb_results,nb_selected)
        print("I selected the %d best solutions I know."%(nb_selected))
        print(type(nb_selected))
        selection=bests.list_results[:nb_selected]
        return Results(selection)
            
    def restrict_to_methods(self,methods):
        if type(methods) is not list:
            methods=[methods]
        New_Results=Results()
        for r in self.list_results:
            if r.method in methods:
                New_Results.add_result(r)
        return New_Results
            
    def test(self, Xtrain, ytrain, Xtest, ytest, nb_test=5):
        nb_test=min(nb_test,self.nb_results)
        l_bests=self.bests(nb=nb_test)
        for r in l_best:
            print('Testing %s with parameters: %s'%(Methods_names[r.method],r.param))
            model=r.method(r.param)
            predictor=model.fit(Xtrain,ytrain)
            score = predictor.score(Xtest, ytest)
            print("Test Score=",score)

    def analyse(self,method='all'):
        if method is not 'all':
            # convert list_results to data frame so that we can run a LogisticRegression on it
            #list=[]
            names_param=list(*(self.list_results[0].param))
            for r in self.list_results:
                print(r.method)
                print(**(r.param))
                print(r.cv_score)
            
            #df=pd.DataFrame(np.array(list).reshape(self.nb_results,len(list[0]), columns = ['method',names_param,'score'])
            #print(df)
    
    
def get_param_choices(method):
    if method is LinearSVC:
        param=[{"penalty":['l2','l1'],
        "loss":['squared_hinge','hinge'],
        "dual":[True,False],
        "tol":[0.0001],
        "C":np.concatenate([np.linspace(0,1,21),np.linspace(1.1,100,21)]),
        "multi_class":['ovr','crammer_singer'],
        "fit_intercept":[True,False],
        "intercept_scaling":[1],
        "class_weight":[None]
        }] 
        
    if method is SVC:
        param=[{"C":np.concatenate([np.linspace(0,1,21),np.linspace(1.1,100,21)]),
            "kernel":['rbf','linear','poly', 'sigmoid', 'precomputed'],
            "degree":range(2,11),
            "gamma":['auto'], 
            "coef0":[0.0], 
            "probability":[False], 
            "shrinking":[True,False], 
            "tol":[0.001],
            "cache_size":[200], 
            "class_weight":[None],
          }]
    if method is LogisticRegression:
        param=[{"penalty":['l2','l1'],
        "dual":[True,False],
        "tol":[0.0001],
        "C":np.linspace(0,2,21),
        "fit_intercept":[True],#,False
        "intercept_scaling":[1],
        "class_weight":[None],
        "random_state":[None],
        "solver":['liblinear','newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        "max_iter":[100],
        "multi_class":['ovr','multinomial'],
          }]
    
    if method is XGBClassifier:
        param=[{"max_depth":range(2,15),
        "learning_rate":[0.1], 
        "n_estimators":[100], #np.linspace(50,500,10)
        "silent":[True], 
        "objective":['multi:softmax'], 
        "booster":['gbtree'], 
        "n_jobs":[4], 
        "nthread":[None], 
        "gamma":[0], 
        "min_child_weight":[1], 
        "max_delta_step":[0], 
        "subsample":[1], 
        "colsample_bytree":[1], 
        "colsample_bylevel":[1], 
        "reg_alpha":[0], 
        "reg_lambda":[1], 
        "scale_pos_weight":[1], 
        "base_score":[0.5]
        }]
    if method not in [LinearSVC, SVC, LogisticRegression, XGBClassifier]:
        print("Method %s not implemented."%method,type(method))

    return param
    
