#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon May  8 09:26:06 2017
@author: batu, ralf
"""

"""
This program implements a Random Forest algorithm for classification.
To change the n_estimators will change the score, please try it.
The programm prints the confusion matrix and
the precision, the recall, and the score.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import gen_sample_all as gs
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def rf(t):
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
    # random forest classifier
    # adapt the parameters to investigate in the behavior 
    clf1=RandomForestClassifier(n_estimators=50,
                                min_samples_split=2,
                                random_state=2,
                                max_features=0.4,
                                oob_score=True,
                                max_depth=4) 
    clf1.fit(X_train,y_train)
    y_pred1=clf1.predict(X_test)
    # compare from score, classification report,cohen kappa, confusion matrix   
    score=accuracy_score(y_test, y_pred1, normalize=False)
    # classification report for 5 classes
    report=classification_report(y_test,
                                 y_pred1,
                                 target_names=['class0','class1',
                                               'class2','class3',
                                               'class4','class5'])
    # calculate cohen_kappa_score
    kappa=cohen_kappa_score(y_test, y_pred1)
    # calculate confusion_matrix
    cnf_matrix = confusion_matrix(y_test, y_pred1,labels=[0,1,2,3,4,5])
    # cross validation part
    predicted = cross_val_predict(clf1, X,y, cv=10)
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
    print("\n")
    print(70*"_")
    
def main():
    report,cnf_matrix,precision,recall=rf(0) # change the selected sample data
    print_report(report,cnf_matrix,precision,recall)

main()
