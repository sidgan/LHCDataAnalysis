import csv as csv
import glob
import pandas as pd 
import numpy as np 
import warnings 
import matplotlib.pyplot as plt

"""
File       : feature_plot.py
Author     : Siddha Ganju <siddhaganju AT gmail dot com>
Description: 

Plot variable frequency against value. This helps to find the correlation between all the 84 parameters and helps in feature selection.  
"""


def main():
    
    path = '/afs/cern.ch/user/s/sganju/private/'
    filename = glob.glob(path + '/*.csv')
    filename_2013 = glob.glob(path + 'dataframe-2013*.csv')
    filename_2014 = glob.glob(path + 'dataframe-2014*.csv')
    filename_2015 = glob.glob(path + 'dataframe-2015*.csv')
   

    def plot_data(filename, hist_label, hist_col):
      data = pd.DataFrame()
      for file_ in filename:
           data_temp = pd.read_csv(file_, index_col=None, header=0)
           data = data.append(data_temp, ignore_index=True)
           plt.hist(np.log(data['naccess'] + 1), alpha=0.5, color = hist_col, label = hist_label)
      plt.show()
    

    plot_data(filename, 'complete data', 'blue')
    plot_data(filename_2013, '2013 data', 'green')
    plot_data(filename_2014, '2014 data', 'red')
    plot_data(filename_2015, '2015 data', 'yellow')
   
 


#main ends here 
if __name__ == '__main__':
	main()	
