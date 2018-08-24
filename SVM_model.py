#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:26:06 2017

@author: batu, ralf
"""

"""
This program implements a Support Vector Machine (SVM) for classification.
To change the n_estimators will change the score, please try it.
The program uses the output of one time step before to make the
estimation more precise. The improvement is great, please try it. 
The programm prints the confusion matrix and the precision, the recall and the score.
"""

# load libaray
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import gen_sample_all as gs
import pickle
from sklearn.externals import joblib

def make_data(t):
    """ generates the data for training """
    path='/home/ralf/pya/Remote-Sensing/'
    X,y=gs.gen_equal_size(0)
    X0=np.zeros((X.shape[0],X.shape[1]+1))
    X0[:,:-1]=X[:,:]
    X0[:,-1]=1.0
    # XR=np.copy(X0)
    if(t==0):
        return X0,y
    X,y=gs.gen_equal_size(1)
    X1=np.zeros((X.shape[0],X.shape[1]+1))
    X1[:,:-1]=X[:,:]
    clf=joblib.load(path+'clf%d.pkl' % 0)
    X1[:,-1]=clf.predict(X0)
    if(t==1):
        return X1,y
    X,y=gs.gen_equal_size(2)
    X2=np.zeros((X.shape[0],X.shape[1]+1))
    X2[:,:-1]=X[:,:]
    clf=joblib.load(path+'clf%d.pkl' % 1)
    X2[:,-1]=clf.predict(X1)
    if(t==2):
        return X2,y
    X,y=gs.gen_equal_size(3)
    X3=np.zeros((X.shape[0],X.shape[1]+1))
    X3[:,:-1]=X[:,:]
    clf=joblib.load(path+'clf%d.pkl' % 2)
    X3[:,-1]=clf.predict(X2)
    return X3,y

def svm1(t):
    """
    uses t as an index to the list of dates and
    calculates:
    the score, the report, the kappa and the confusion matrix
    it returns the report and the confusion matrix only
    """
    path='/home/ralf/pya/Remote-Sensing/'
    X,y=make_data(t)
    #print(X[0:5,:])
    # spilt the dataset into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)
    # define a SVM with the parameters C and gamma and
    # a RBF kernel (try also a  ‘linear’, ‘poly’, ‘sigmoid’)
    classifier = svm.SVC(kernel='rbf', C=100, gamma=10.0)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    #clf = svm.SVC(kernel='rbf', C=1000, gamma=10.0)
    #clf.fit(X,y)
    # y_a = classifier.fit(X, y).predict(X)
    # compare from score, classification report,cohen kappa, confusion matrix
    # calculate accuracy_score 
    # score=accuracy_score(y_test, y_pred, normalize=False)
    # classification report for 5 classes
    report=classification_report(y_test,
                                 y_pred,
                                 target_names=['class0','class1',
                                               'class2','class3',
                                               'class4','class5'])
    # calculate confusion_matrix we use a model with 5 classes!
    cnf_matrix = confusion_matrix(y_test, y_pred,labels=[0,1,2,3,4,5])
    # cnf_matrix1 = confusion_matrix(y, y_a,labels=[0,1,2,3,4,5])
    joblib.dump(classifier,path+'clf%d.pkl' % t)
    # cross validation part
    predicted = cross_val_predict(classifier, X,y, cv=10)
    # print(predicted)
    #print(y)
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
    report,cnf_matrix,precision,recall=svm1(0) # change the selected sample data
    print_report(report,cnf_matrix,precision,recall)
    print("\n")
    print(70*"_")
    report,cnf_matrix,precision,recall=svm1(1) # change the selected sample data
    print_report(report,cnf_matrix,precision,recall)
    print("\n")
    print(70*"_")
    report,cnf_matrix,precision,recall=svm1(2) # change the selected sample data
    print_report(report,cnf_matrix,precision,recall)
    print("\n")
    print(70*"_")
    report,cnf_matrix,precision,recall=svm1(3) # change the selected sample data
    print_report(report,cnf_matrix,precision,recall)
    print("\n")
    print(70*"_")

main()
