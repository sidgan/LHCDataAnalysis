"""
File       : feature_plot.py
Author     : Siddha Ganju <siddhaganju AT gmail dot com>
Description: 

log plots for applying cuts  
"""


import random
import csv as csv
import glob
import pandas as pd 
import numpy as np 
import pylab as P
import warnings 
from sklearn.cross_validation import train_test_split 
import matplotlib.pyplot as plt

def main():
    
    path = '/afs/cern.ch/user/s/sganju/private/'
    filename = glob.glob(path + '/*.csv')
    filename_2013 = glob.glob(path + 'dataframe-2013*.csv')
    filename_2014 = glob.glob(path + 'dataframe-2014*.csv')
    filename_2015 = glob.glob(path + 'dataframe-2015*.csv')
    data = pd.DataFrame()
    for file_ in filename_2013:
	data_temp = pd.read_csv(file_, index_col=None, header=0)
        data = data.append(data_temp, ignore_index=True)
    plt.hist(np.log(data['totcpu'] + 1), alpha=0.5, color = 'green' , label = '2013 data') 
    data = pd.DataFrame()   
    for file_ in filename_2014:
	data_temp = pd.read_csv(file_, index_col=None, header=0)
        data = data.append(data_temp, ignore_index=True)
    plt.hist(np.log(data['totcpu'] + 1), alpha=0.5, color =  'red', label = '2014 data')
    data = pd.DataFrame()
    for file_ in filename_2015:
	data_temp = pd.read_csv(file_, index_col=None, header=0)
        data = data.append(data_temp, ignore_index=True)
    plt.hist(np.log(data['totcpu'] + 1), alpha=0.5, color = 'yellow', label = '2015 data')

    

    data = pd.DataFrame()
    for file_ in filename_2013:
	data_temp = pd.read_csv(file_, index_col=None, header=0)
        data = data.append(data_temp, ignore_index=True)
    plt.hist(np.log(data['naccess'] + 1), alpha=0.5, color = 'blue' , label = '2013 data') 
    data = pd.DataFrame()   
    for file_ in filename_2014:
	data_temp = pd.read_csv(file_, index_col=None, header=0)
        data = data.append(data_temp, ignore_index=True)
    plt.hist(np.log(data['naccess'] + 1), alpha=0.5, color =  'orange', label = '2014 data')
    data = pd.DataFrame()
    for file_ in filename_2015:
	data_temp = pd.read_csv(file_, index_col=None, header=0)
        data = data.append(data_temp, ignore_index=True)
    plt.hist(np.log(data['naccess'] + 1), alpha=0.5, color = 'pink', label = '2015 data')

    

    data = pd.DataFrame()
    for file_ in filename_2013:
	data_temp = pd.read_csv(file_, index_col=None, header=0)
        data = data.append(data_temp, ignore_index=True)
    plt.hist(np.log(data['nusers'] + 1), alpha=0.5, color = 'black' , label = '2013 data') 
    data = pd.DataFrame()   
    for file_ in filename_2014:
	data_temp = pd.read_csv(file_, index_col=None, header=0)
        data = data.append(data_temp, ignore_index=True)
    plt.hist(np.log(data['nusers'] + 1), alpha=0.5, color =  'cyan', label = '2014 data')
    data = pd.DataFrame()
    for file_ in filename_2015:
	data_temp = pd.read_csv(file_, index_col=None, header=0)
        data = data.append(data_temp, ignore_index=True)
    plt.hist(np.log(data['nusers'] + 1), alpha=0.5, color = 'magenta', label = '2015 data')

    plt.show()


#main ends here 
if __name__ == '__main__':
	main()	
