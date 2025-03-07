#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:26:06 2017

@author: batu, ralf
"""

"""
This program implements a Support Vector Machine (SVM) for classification.
To change the n_estimators will change the score, please try it.
The programm prints the confusion matrix and
the precision, the recall and the score.
"""

# load libaray
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import svm
import matplotlib.pyplot as plt
import gen_sample_all as gs

"""
precision: How many selected items are relevant? TP/(TP+FP)
recall:    How many relevant items are selected? TP/(TP+FN)
"""

def optimize(t):
    """ GridSearchCV """
    # load data using gen_sample_all
    X,y=gs.gen_data(t)
    # spilt the dataset into training data and test data
    kfold=StratifiedKFold(n_splits=10,shuffle=True)
    model=svm.SVC()
    C=[1,10,100,500,1000]
    gamma=[0.2,0.5,1.0,2.0,5.0,10.0]
    kernel=['linear', 'poly', 'rbf', 'sigmoid']
    params={'kernel' : kernel,
            'C' : C,
            'gamma' : gamma}
    print(params)
    grid=GridSearchCV(model,params,cv=kfold)
    grid_res=grid.fit(X,y)
    print("Params:", grid_res.best_params_)
    return grid_res


def svm1(t,best):
    """
    uses t as an index to the list of dates and
    calculates:
    the score, the report, the kappa and the confusion matrix
    it returns the report and the confusion matrix only
    """
    # load data using gen_sample_all
    X,y=gs.gen_data(t)
    # spilt the dataset into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    # define a SVM with the parameters C and gamma and
    # a RBF kernel (try also a  ‘linear’, ‘poly’, ‘sigmoid’)
    if(best is None):
        classifier = svm.SVC(kernel='rbf', C=1000,gamma=1.0)
    else:  
        classifier = svm.SVC(kernel=best['kernel'],gamma=best['gamma'],C=best['C'])
    # classifier = svm.SVC(kernel='poly', C=100,gamma=10.0)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    # compare from score, classification report,cohen kappa, confusion matrix
    # calculate accuracy_score 
    score=accuracy_score(y_test, y_pred, normalize=False)
    # classification report for 5 classes
    report=classification_report(y_test,
                                 y_pred,
                                 target_names=['class0','class1',
                                               'class2','class3',
                                               'class4'])
    # calculate cohen_kappa_score
    kappa=cohen_kappa_score(y_test, y_pred)
    # calculate confusion_matrix we use a model with 5 classes!
    cnf_matrix = confusion_matrix(y_test, y_pred,labels=[0,1,2,3,4])
    # cross validation part
    predicted = cross_val_predict(classifier, X,y, cv=10)
    precision=precision_score(y, predicted,average=None)
    recall=recall_score(y, predicted,average=None)
    return report, cnf_matrix, precision, recall

    
def print_report(report,cnf_matrix,precision,recall):
    """ 
    prints the score, the report,the kappa and the matrix
    """
    print(70*"_")
    print("\n")
    print(report)
    print(70*"_")
    print("\n")
    print("confusion matrix: \n")
    print(cnf_matrix)
    print("\n")
    print(70*"_")
    print("\n")
    print('precision: mean={0:.3f} std={1:.3f}'.format(np.mean(precision),
                                                       np.std(precision)))
    print('recall: mean={0:.3f} std={1:.3f}'.format(np.mean(recall),
                                                    np.std(recall)))
    

def main():
    t=2
    best=None
    best=optimize(t).best_params_
    report,cnf_matrix,precision,recall=svm1(t,best) # change the selected sample data
    print_report(report,cnf_matrix,precision,recall)
    print("\n")
    print(70*"_")

main()
