#!/usr/bin/env python3

# import some Python modules 
import pandas as pd
import numpy as np
import copy
import sys

""" 
Reads data from the EXCEL-Sheets for each fruit and generates a sample
from it 
"""
__Print__=False

class zbell(object):
    """ reads the data from Bernd Zbell and allows to select
        it according to the crop, the wave and the date
        Please adapt the path according your needs!
    """
    def __init__(self):
        self.path='./'
        self.files={'ALF':'2002_HYP2_MCA12_4501050_SUPER_ALF.xls',
                    'TRI':'2002_HYP2_MCA12_4501050_SUPER_TRI.xls',
                    'COC':'2002_HYP2_MCA12_4501050_SUPER_COC.xls',
                    'WABR':'2002_HYP2_MCA12_4501050_SUPER_WBAR.xls',
                    'LUP':'2002_HYP2_MCA12_4501050_SUPER_LUP.xls',
                    'WPR':'2002_HYP2_MCA12_4501050_SUPER_WRP.xls',
                    'PEA':'2002_HYP2_MCA12_4501050_SUPER_PEA.xls',
                    'WRY':'2002_HYP2_MCA12_4501050_SUPER_WRY.xls',
                    'POT':'2002_HYP2_MCA12_4501050_SUPER_POT.xls',
                    'WWH':'2002_HYP2_MCA12_4501050_SUPER_WWH.xls',
                    'SMA':'2002_HYP2_MCA12_4501050_SUPER_SMA.xls'}
        self.waves=[490,560,665,705,740,783,842,865,1610,2190]
        self.crop=None
        self.file=None
        self.tb=None
        return
    def use(self,crop):
        """ reads the Excel Sheet for a crop """
        if(crop in self.files):
            self.crop=crop
            self.file=self.files[crop]
            self.tb=pd.read_excel(self.path+self.file,header=[0,1],sheetname='HYP2')
    def dates(self):
        """ returns the list of dates for the selected crop """
        if(self.tb is None):
            return None
        return self.tb.columns.levels[0]
    def select(self,date_ix,wave):
        """ 
        returns a list of values for a date_ix and an wave
        the date_ix is the index of the dates from get_dates()
        """
        if(self.tb is not None and wave in self.waves and 
            date_ix in range(len(self.tb.columns.levels[0]))):
            return self.tb.ix[wave][self.tb.columns.levels[0][date_ix]].values
        return None
        
""" 
The most important part of this script:
it allows to combine different crops to use it as training data for the
machine learning algorithms.
Some examples are given below
"""

#modell={'WWH':zbell(), 'WPR':zbell(), 'WRY':zbell(), 'SMA':zbell()}
modell={'WWH':zbell(), 'WPR':zbell(), 'WRY':zbell(), 'SMA':zbell(),'POT':zbell()}
#modell={'WWH':zbell(), 'WPR':zbell(), 'WRY':zbell()}

# load the modell
for key in modell.keys():
    modell[key].use(key)
    print(key,modell[key].dates())
 

def gen_data(dix):
    """ 
    define a function to collect the data for all crops and all waves at
    a given date.
    returns a numpy array with the data and a target array with 
    ids=[0,1,2,...] for the used crops
    """
    # define the length of the data according to Sentinel2
    waves=[490,560,665,705,740,783,842,865,1610,2190]
    lx={}
    for key in modell.keys():
        lx[key]=0
        for wave in modell[key].waves:
            if(lx[key]==0):
                lx[key]=len(modell[key].select(dix,wave))
            else:
                lx[key]=min(lx[key],len(modell[key].select(dix,wave)))
    print(lx)
    print()
    # define the waves to store the different waves
    w={}
    for wave in waves:
        w[wave]=np.array([])
    #define the result table
    target=np.array([])
    data=np.zeros((len(waves),sum(lx.values())))
              
    # fill the waves and target
    v=0
    for key in lx.keys():
        # print(key)
        target=np.append(target,[v for k in range(lx[key])])
        v+=1
        
    for i,key in enumerate(modell.keys()):   
        for j,wave in enumerate(modell[key].waves):
            if(__Print__==True):
                print(key,dix,wave,modell[key].select(dix,wave))
            w[wave]=np.append(w[wave],modell[key].select(dix,wave))
            
    # fill the result data
    i=0
    for key in waves:
        if(__Print__==True):
            print(i,key)
        data[i]=w[key]
        i+=1
    return data.T,np.array(target)
          
        
    
