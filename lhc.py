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
	#read one file for example
	#removing popular params from training set 

	#plots to check popularity metric

	#data = train.drop(train.columns[[8, 15, 82]], axis=True, inplace=True) #removed target values 
    
    path = '/home/sidgan/Downloads/'
    filename = glob.glob(path + '/*.csv')
    filename_2013 = glob.glob(path + 'dataframe-2013*')
    filename_2014 = glob.glob(path + 'dataframe-2014*')
    filename_2015 = glob.glob(path + 'dataframe-2015*')
    data = pd.DataFrame()

    

    def plot_data(filename):
      for file_ in list(filename):
           data_temp = pd.read_csv(file_, index_col=None, header=0)
           data = data.append(data_temp, ignore_index=True)
      plt.hist(np.log(data['naccess'] + 1))
      plt.show()
    

    plot_data(filename)
    plot_data(filename_2013)
    plot_data(filename_2014)
    plot_data(filename_2015)

if __name__ == '__main__':
    main()
