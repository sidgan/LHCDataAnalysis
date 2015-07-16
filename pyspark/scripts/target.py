import csv as csv
import glob
import pandas as pd 
import numpy as np 
import warnings 

def main():
    
    path = '/afs/cern.ch/user/s/sganju/private/2013.csv'
    data = pd.read_csv(path, index_col=None, header=0)
    
    data['popular'] = 0 
    print data.head()
    for each_row in data:
      condition_naccess = data['naccess'].map(lambda x: 1 if x > 10 else 0) 
      condition_totcpu = data['totcpu'].map(lambda x: 1 if x > 10 else 0) 
      condition_nusers = data['nusers'].map(lambda x: 1 if x > 5 else 0) 
      condition = condition_naccess && condition_totcpu && condition_nusers
      data['popular'] = condition
           


#main ends here 
if __name__ == '__main__':
	main()	
