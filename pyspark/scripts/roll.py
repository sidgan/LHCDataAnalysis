#!/bin/bash
#!/usr/bin/env python
#author sidgan 

import stdlib 
import subprocess
from subprocess import call 
import csv as csv
import glob
import pandas as pd 
import numpy as np 
import warnings 
import matplotlib.pyplot as plt


"""
File       : roll.py
Author     : Siddha Ganju <siddhaganju AT gmail dot com>
Description: 

#roll over 2013 add each week and perform rolling forecasts for each week 
 
"""

def main():
    
    total_filename = "dataframe-20130108-20130114.csv" 
    #start with the first week        

    def combine_files(filename, year):
    	  #data = pd.DataFrame()
          for each_filename in filename:
            #add to total_filename 
            total_filename.append(each_filename) 
            #convert to classification problem
            call(["python classification_target.py", "--fin each_filename", "--fout each_filename", "--target=TARGET", "--verbose"])
            #merge to total.csv 
            call(["python merge_csv", "--fin ", "--fout total.csv", "--verbose"])
            #run 4 models on total_filename 
            call(["python pipeline.py", "--fin", "--fout", "--verbose"])
            #got predictions as output for each model 
            #verify predictions
            #calculate f1, recall, accuracy, precision in a separate file
            call(["check_predictions", "--fin ", "--fpred" , "scorer " ])
            #apply corrections from predictions 
            
  
#main ends here 
if __name__ == '__main__':
	main()	
