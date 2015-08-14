#!bin\bash
#author sidgan 

"""
File       : feature_plot.py
Author     : Siddha Ganju <siddhaganju AT gmail dot com>
Description: 

Spark machine learning functionality 

"""


from __future__ import print_function
import sys
#import spark files 
from pyspark.mllib.classification import LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import  LinearRegressionWithSGD
from pyspark import SparkContext, SparkConf

#for reading csv files
import pyspark_csv as pycsv

#for a python classifier import scikit-learn 
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import datasets
from sklearn import utils 
from sklearn.utils import shuffle
from multiprocessing import Process
import datetime
import resource
import os
#from mpi4py import MPI
import resource 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import time 
from sklearn.naive_bayes import GaussianNB
import time
import csv as csv
import glob
import pandas as pd
import psutil 
import numpy as np 
import sklearn 
import sklearn.cluster 
import optparse 
from pandas import * 
from pandas import DataFrame
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation 
from sklearn.metrics import precision_recall_curve
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm 
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score



#add spark content 
sc = SparkContext(appName="sidgan")
sc.addPyFile('/home/spark-1.4.1-bin-hadoop2.4/pyspark-csv/pyspark_csv.py')

def transform_csv():
    global data
    global target
    #make target column for classification
    #target = data.map(convert)
    #this gives RDD
    #target should be float and not RDD

    target = filter(convert)
    #should give target as a float 

def merge_csv():
    global data
    global week 
    #merge RDD to implement incremental growth
    #call rdd persist method
    data.join(week)
    data.persist()
    #save data for future use 


def algo(a):
    global data
    global week 
    global target
    test = week 
    week_target = week.map(convert)
    #apply(convert, axis=1)
    #np.random.seed(123)
    data_final = LabeledPoint(target, data)
    #make rdd that is input for algo 


    if a == 'sgd':
        #time_0 = time.time()
        lrm = LinearRegressionWithSGD.train(sc.parallelize(data_final), iterations=10, initialWeights=np.array([1.0]))
        print (abs(lrm.predict(test)))
        print time.time() - time_0 
       

def convert(df):
    threshold_naccess = 10 
    threshold_nusers = 5 
    return  df['naccess'] > threshold_naccess and df['nusers'] > threshold_nusers

def cal_score(method, clf, features_test, target_test):
    #target_test = target_test.tolist()
    perf_measure( features_test, target_test)

def perf_measure(target, predictions):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(target)): 
        if predictions[i]==target[i]==1:
            TP += 1
    for i in range(len(target)): 
        if target[i]==1 and target[i]!=predictions[i]:
            FP += 1
    for i in range(len(target)): 
        if target[i]==predictions[i]==0:
            TN += 1
    for i in range(len(target)): 
        if target[i]==0 and target[i]!=predictions[i]:
            FN += 1
    print ('Accuracy Score ', TP,FP,TN,FN)
    

def monitor():
    while True:
        cpu_data = psutil.cpu_percent(interval=1)
        mem_data = psutil.virtual_memory().used
        print (cpu_data, mem_data, psutil.virtual_memory().active)  
        time.sleep(1)

def main():
    global data        
    global week
    p = optparse.OptionParser()
    #take inputs 
    #take training data set 
    p.add_option('--train_dataset', '-i', default='/afs/cern.ch/user/s/sganju/private/2013.csv')
    #specify target column
    p.add_option('--target', '-y', default="target")
    #add different algos 
    #random forest 
    p.add_option('--algo', '-a',default = "sgd")
    #parse inputs
    #read options
    options, arguments = p.parse_args()
    a = options.algo 
    #hdfs path is the new path 
    path =sys.argv[1] #hdfs://samrouch-mesos-01:54310/user/root/test/"
    #know all of 2014 data files 
    #use glob 
    filename_2014 = path + 'dataframe-20140101-20140107.csv'
        
    #data = sc.textFile(path + "2013.csv")
    #data=sc.textFile(path+"2013.csv").map(lambda line: line.split(",")).filter(lambda line: len(line)>1).map(lambda line: (line[0],line[1])).collect()
    #read csv file using pycsv
    plaintext_rdd = sc.textFile(path+'2013.csv')
    from pyspark.sql import SQLContext
    sqlCtx = SQLContext(sc)    
    data = pycsv.csvToDataFrame(sqlCtx, plaintext_rdd)

    for each_file in filename_2014:
        week = sc.textFile(each_file)
        print (' ==== ', each_file )
        transform_csv()
        algo(a)
	    merge_csv() 
 	break 
         
    
#main ends here 
if __name__ == '__main__':
        proc = Process(target = monitor)
        proc.daemon = True 
        proc.start()
        #time0 = time.time()
        main()
        print ' === ', time.time() - time0 , ' : Total Time for execution'
        print ' === Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
