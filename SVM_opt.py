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
import gen_sample_all as gen

X0,y0=gen.gen_data(0)
X1,y1=gen.gen_data(1)
X2,y2=gen.gen_data(2)
X3,y3=gen.gen_data(3)

from sklearn.preprocessing import OneHotEncoder
def data_for_training(nr):
    """ nr: 0,1,2,3 using X0,X1,X2,X3 """
    global X0,X1,X2,X3, y0,y1,y2,y3
    size=min(X0.shape[0],X1.shape[0],X2.shape[0],X3.shape[0])
    spec=X0.shape[1] # lenght of used spectral data
    Xtrain=np.zeros((size,nr+1,spec)) 
    ytrain=np.zeros((size,nr+1,1)) # without encoding
    enc_size=int(np.max(y0))+1
    enc=OneHotEncoder([enc_size])
    y_features=[]
    for i in range(enc_size):
        y_features.append([i])
    print(y_features)
    enc.fit(y_features)
    yt=np.zeros((size,nr+1,enc_size)) # with encoding
    k=0
    while(k<size):
        i=np.random.randint(0,y0.shape[0])
        Xtrain[k,0,:]=X0[i,:]
        ytrain[k,0]=y0[i] # for example wheat
        yt[k,0,:]=enc.transform(int(y0[i])).toarray()[0]
        k+=1
    if nr<1:
        return Xtrain,yt,ytrain
    k=0 
    while(k<size):
        sel=ytrain[k,0,0]
        # print('1',k,sel)
        i=np.random.randint(0,y1.shape[0])
        while(y1[i]!=sel):
            i=np.random.randint(0,y1.shape[0])
        Xtrain[k,1,:]=X1[i,:]
        ytrain[k,1]=sel
        yt[k,1,:]=enc.transform(int(sel)).toarray()[0]
        k+=1
    if nr<2:
        return Xtrain,yt,ytrain
    k=0
    while(k<size):
        sel=ytrain[k,1,0]
        # print('1',k,sel)
        i=np.random.randint(0,y2.shape[0])
        while(y2[i]!=sel):
            i=np.random.randint(0,y2.shape[0])
        Xtrain[k,2,:]=X2[i,:]
        ytrain[k,2]=sel
        yt[k,2,:]=enc.transform(int(sel)).toarray()[0]
        k+=1
    if nr<3:
        return Xtrain,yt,ytrain
    k=0
    while(k<size):
        sel=ytrain[k,2,0]
        # print('1',k,sel)
        i=np.random.randint(0,y3.shape[0])
        while(y3[i]!=sel):
            i=np.random.randint(0,y3.shape[0])
        Xtrain[k,3,:]=X3[i,:]
        ytrain[k,3]=sel
        yt[k,3,:]=enc.transform(int(sel)).toarray()[0]
        k+=1
    return Xtrain,yt,ytrain

def svm1(t):
    """
    uses t as an index to the list of dates and
    calculates:
    the score, the report, the kappa and the confusion matrix
    it returns the report and the confusion matrix only
    """
    # load data using gen_sample_all
    #X,y=gs.gen_data(t)
    path='/datadisk/pya/Remote-Sensing/'
    
    #X,y=gs.gen_data(t)
    XL,YYY,yL=data_for_training(t)
    if(t==0):
        X=XL[:,0,:]
        y=yL[:,0,0]
    else:
        try:
            y=yL[:,t,0]
            p=np.load(path+'S%d.npy' % (t-1))
            print(p)
            X=np.zeros((XL.shape[0],XL.shape[2]+1))
            switch=X.shape[1]-1
            for i in range(X.shape[0]):
                X[i,:switch]=XL[i,t,:]
                X[i,switch]=p[int(y[i])]
        except:
            print('could not open: S%d.npy' % (t-1))
    # spilt the dataset into training data and test data
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    X_train=X
    X_test=X
    y_train=y
    y_test=y
    # define a SVM with the parameters C and gamma and
    # a RBF kernel (try also a  ‘linear’, ‘poly’, ‘sigmoid’)
    classifier = svm.SVC(kernel='rbf', C=10,gamma=1.0) # C=1000 is the best
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    y_pred1 = classifier.fit(X_train, y_train).predict(X_train)
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
    cnf_matrix1 = confusion_matrix(y_train, y_pred1,labels=[0,1,2,3,4])
    np.save(path+'S%d.npy' % t, np.diagonal(cnf_matrix1)/np.sum(np.diagonal(cnf_matrix1)))
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

main()
