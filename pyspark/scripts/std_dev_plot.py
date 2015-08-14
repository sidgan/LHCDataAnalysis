#!bin\bash
#author sidgan 

"""
File       : std_dev_plot.py
Author     : Siddha Ganju <siddhaganju AT gmail dot com>
Description: 
Plots for standard deviation

"""

import csv as csv
import glob
import pandas as pd
import numpy as np 
from pandas import * 
from pandas import DataFrame


data = pd.DataFrame()
week = pd.DataFrame()
data_target = pd.DataFrame()
data_target['week'] = 0
data_target['tp'] = 0
data_target['tn'] = 0 


def convert(df):
    threshold_naccess = 10 
    threshold_nusers = 5 
    return df['naccess'] > threshold_naccess and df['nusers'] > threshold_nusers

def main():
    global data        
    global week
    path = '/afs/cern.ch/user/s/sganju/private/'
    filename_2014 = glob.glob(path + 'dataframe-2014*.csv')
    for each_file in filename_2014:
        week = pd.read_csv(each_file)
        week = week[['naccess', 'nusers']]
        week['target'] = 0
        week['target'] = week.apply(convert, axis=1)
        week['target'] = week['target'].astype(int)
        #print week['target']
        tp = 0 
        tn = 0 
        for row in week['target']:
            if row  == 1 :
                tp = tp + 1 
            else : 
	        #print 'or here'
                #print  '--'
                tn = tn + 1 
                #calclate tp and tn by adding ones and zeroes
        #append to target data set
        #data_target['week']['row']
	print tp , tn , each_file 
	
 
         



    
#main ends here 
if __name__ == '__main__':
	main()
	
