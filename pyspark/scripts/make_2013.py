"""
File       : make_2013.py
Author     : Siddha Ganju <siddhaganju AT gmail dot com>
Description: 

Combine files of a single year 
"""

import csv as csv
import glob
import pandas as pd 

def main():
    
    path = '/afs/cern.ch/user/s/sganju/private/'
    filename_2013 = glob.glob(path + 'dataframe-2013*.csv')

 
    def combine_files(filename, year):
    	  data = pd.DataFrame()
  	  for file_ in filename:
        	   data_temp = pd.read_csv(file_, index_col=None, header=0)
        	   data = data.append(data_temp, ignore_index=True)
   	  data.to_csv( (year + '.csv') , sep=',')  
  
    combine_files(filename_2013, '2013')

  
    
#main ends here 
if __name__ == '__main__':
	main()	
