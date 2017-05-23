#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:26:06 2017

@author: batu, ralf
"""

"""
This program implements a Adaboost algorithm for classification.
Adaboost uses a regression tree as basis algorithm.
To change the n_estimators will change the score, please try it.
The programm prints the confusion matrix and
the precision, the recall and the score.
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import sys  
sys.path.append('/datadisk/Remote-Sensing')  # adapt this!
import gen_sample_all as gs
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def ada(t):
    """
    uses t as an index to the list of dates and
    calculates:
    the score, the report, the kappa and the confusion matrix
    it returns the report and the confusion matrix only
    """
    # load data using gen_sample_all
    X,y=gs.gen_data(t)
    # spilt the dataset into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # adaboost classifier based on decision tree
    # adpat the parameters max_depth and n_estimators please
    clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=4),n_estimators=50)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    
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
    return report, cnf_matrix
    
def print_report(report,cnf_matrix):
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

def main():
    report,cnf_matrix=ada(2) # change the selected sample data
    print_report(report,cnf_matrix)

main()
    
