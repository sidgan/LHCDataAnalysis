import csv as csv
import glob
import pandas as pd 

def main():
	p = optparse.OptionParser()
	#take inputs 
	#input file in which target is to be added 
	p.add_option('--input_file', '-i', default='', type='string')
	#output file with target column
	p.add_option('--output_file', '-o', default='', type = 'string')
	#read options
	options, arguments = p.parse_args()

	#input_stream = fopen(p.i, 'r') #input file
	#convert to pandas dataframe 
	#data = pd.DataFrame()
	data = pd.read_csv(p.i)
  	#data = pd.DataFrame()
  	for file_ in filename:
           data_temp = pd.read_csv(file_, index_col=None, header=0)
           data = data.append(data_temp, ignore_index=True)
        data.to_csv( (year + '.csv') , sep=',')  
  
    
#main ends here 
if __name__ == '__main__':
	main()	
