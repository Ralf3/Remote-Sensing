#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:58:52 2018

@author: ralf
"""

""" LSTM using the remote sensing data set """
""" fivenum implementation using numpy """

import numpy as np
import pylab as plt
import sys
sys.path.append('/datadisk/Remote-Sensing')
import gen_sample_all as gen

def sevennum(moons):
    """ help function often useful 
        read a list of objects and calculates some precentiles
    """
    return(
            [np.min(moons),
             np.percentile(moons, 10, interpolation='midpoint'),
             np.percentile(moons, 25, interpolation='midpoint'),
             np.median(moons),
             np.percentile(moons, 75, interpolation='midpoint'),
             np.percentile(moons, 90, interpolation='midpoint'),
             np.max(moons)]
            )

X0,y0=gen.gen_data(0)
X1,y1=gen.gen_data(1)
X2,y2=gen.gen_data(2)
X3,y3=gen.gen_data(3)


""" organize the data using random sort """
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


""" import layers LSTM and Dense plus Sequential """
from keras.layers import LSTM
from keras.layers import Dense
# from keras.layers import TimeDistributed
from keras.models import Sequential

""" make the data """
sample=min(X0.shape[0],X1.shape[0],X2.shape[0],X3.shape[0]) # the sample for all
time_steps=4    # number of steps 
features=10     # spectral componentes
enc_size=5      # five dense nodes for each 
Xtrain,ytrain,ytest=data_for_training(time_steps-1)
""" define the model """
model=Sequential()
model.add(LSTM(enc_size,return_sequences=True, input_shape=(time_steps,features)))
#model.add(LSTM(enc_size,return_sequences=True))
model.add(Dense(enc_size,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
# train the model
model.fit(Xtrain[0:70,:,:],ytrain[0:70,:,:],epochs=800,batch_size=10,validation_data=(Xtrain[70:,:,:],ytrain[70:,:,:]))
res=model.predict(Xtrain)
res1=[]
for i in range(res.shape[0]):
    res1.append(np.argmax(res[i],axis=1))

def confusion(pos):
    """ calculates the confusion matrix for the pos=0,1,2,3 """
    global ytrain,res1
    time_steps=res1[0].shape[0] # time_steps
    enc_size=ytrain.shape[-1]   # number of classes
    
    if(pos<0 or pos>time_steps):
        return None
    conf_mat=np.zeros((enc_size,enc_size))
    for i in range(ytrain.shape[0]):
        conf_mat[int(np.argmax(ytrain[i])),int(res1[i][pos])]+=1
    return conf_mat

print(confusion(0))
print(confusion(1))
print(confusion(2))
print(confusion(3))

