#!bin/bash
#author sidgan

#generate the target values

#if naccess > 10
# if nusers > 5
#then keep
#values obtained from histograms

import optparse
import csv as csv
import pandas as pd 

def main():
	p = optparse.OptionParser()
	#take inputs 
	p.add_option('--input_file', '-i', default='', type='string')
	threshold_nusers = 5
	threshold_naccess = 10
	#read options
	options, arguments = p.parse_args()
	data = pd.read_csv(options.input_file)
	#add target column to data 
	data['target'] = 0
	#transformation to classficiation problem
	def convert(df):
		return df['naccess'] > threshold_naccess and df['nusers'] > threshold_nusers
	#perform conversion
	data['target'] = data.apply(convert, axis=1)
	data['target'] = data['target'].astype(int)
	data.to_csv(options.input_file)
	#write target col to file

if __name__ == '__main__':
    main()
