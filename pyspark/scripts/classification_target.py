#!bin/bash

#author sidgan

#generate the target values

#if naccess > 10 then keep it 
# if nusers > 5


import optparse
import csv as csv
import pandas as pd 
import numpy as np 

def main():
	p = optparse.OptionParser()
	#take inputs 
	#input file in which target is to be added 
	p.add_option('--input_file', '-i', default='', type='string')
	#output file with target column
	p.add_option('--output_file', '-o', default='', type = 'string')
	threshold_nusers = 5
	threshold_naccess = 10
	#read options
	options, arguments = p.parse_args()

	#convert to pandas dataframe 
	#data = pd.DataFrame()
	data = pd.read_csv(p.i)
	#add target col to data 
	data['target'] = 0
	#open output file 
	#transformation to classficiation problem
	#generate target values

 	#read each line 
    #check value for naccess and nusers
    #if > threshold then write target as one

    for each_line in data:
    	if (each_line['naccess'] > threshold_naccess && each_line['nusers'] > threshold_nusers )
    	{
    		each_line['target'] = 1 
    		#popular 
    	}
    	else
    	{
    		each_line['target'] = 0 
    		#not popular 
    	}

   
   
if __name__ == '__main__':
    main()
