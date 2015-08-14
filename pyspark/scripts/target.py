#!/bin/bash
#!/usr/bin/env python
#author sidgan 

import csv as csv
import glob
import pandas as pd 
import numpy as np 
import warnings 


"""
File       : target.py
Author     : Siddha Ganju <siddhaganju AT gmail dot com>
Description: 

generate the target values for classification

if naccess > 10 AND  nusers > 5, then keep values obtained 

This file has a new version. Use classification_target.py
 
"""

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
