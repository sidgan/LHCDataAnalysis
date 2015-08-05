#!bin\bash
#author sidgan 

from multiprocessing import Process
import datetime
import resource
import os
import resource 
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
from sklearn.cross_validation import cross_val_score

data = pd.DataFrame()
week = pd.DataFrame()

def transform_csv():
    global data
    data = data[["id", "cpu", "creator", "dbs" , "dtype" , "era" ,  "nblk" , "nevt" , "nfiles" , "nlumis" , "nrel" , "nsites" , "nusers" , "parent" , "primds" , "proc_evts" , "procds" , "rnaccess" , "rnusers" , "rtotcpu" , "size" , "tier" , "totcpu" , "wct", "naccess"]]
    data['target'] = 0
    data['target'] = data.apply(convert, axis=1)
    data['target'] = data['target'].astype(int)
    
def merge_csv():
    global data
    global week 
    data = data[["id", "cpu", "creator", "dbs" , "dtype" , "era" ,  "nblk" , "nevt" , "nfiles" , "nlumis" , "nrel" , "nsites" , "nusers" , "parent" , "primds" , "proc_evts" , "procds" , "rnaccess" , "rnusers" , "rtotcpu" , "size" , "tier" , "totcpu" , "wct", "naccess"]]
    data = concat([data, week])

def algo(a):
    global data
    global week 
    target = data['target']
    data = data[["id", "cpu", "creator", "dbs" , "dtype" , "era" ,  "nblk" , "nevt" , "nfiles" , "nlumis" , "nrel" , "nsites" , "nusers" , "parent" , "primds" , "proc_evts" , "procds" , "rnaccess" , "rnusers" , "rtotcpu" , "size" , "tier" , "totcpu" , "wct", "naccess"]]
    week['target'] = 0
    week['target'] = week.apply(convert, axis=1)
    week['target'] = week['target'].astype(int)
    test1 = week
    week = week[["id", "cpu", "creator", "dbs" , "dtype" , "era" ,  "nblk" , "nevt" , "nfiles" , "nlumis" , "nrel" , "nsites" , "nusers" , "parent" , "primds" , "proc_evts" , "procds" , "rnaccess" , "rnusers" , "rtotcpu" , "size" , "tier" , "totcpu" , "wct", "naccess"]]
    if a == 'rf':
        #RANDOM FOREST CLASSIFIER 
        rf = RandomForestClassifier(n_estimators=100)
        rf = rf.fit(data, target)
	predictions = rf.predict(week)
	cal_score("RANDOM FOREST", rf, predictions, test1['target'])
    if a == "sgd":
        #SGD CLASSIFIER     
        clf = SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
            fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
            loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
            random_state=None, shuffle=True, verbose=0,
            warm_start=False)
        clf.fit(data, target)
        predictions = clf.predict(week)
	cal_score("SGD Regression",clf, predictions, test1['target'])
    if a == "nb":
	clf = GaussianNB()
	clf.fit(data, target)
	predictions = clf.predict(week)
	cal_score("NAIVE BAYES", clf, predictions, test1['target'])

def convert(df):
    threshold_naccess = 10 
    threshold_nusers = 5 
    return df['naccess'] > threshold_naccess and df['nusers'] > threshold_nusers

def cal_score(method, clf, features_test, target_test):
    target_test = target_test.tolist()
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
    print '+++ Accuracy Score ', TP,FP,TN,FN
    

def monitor():
    while True:
	cpu_data = psutil.cpu_percent(interval=1)
	mem_data = psutil.virtual_memory().used
	print cpu_data, mem_data, psutil.virtual_memory().active  
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
    p.add_option('--algo', '-a',default = "rf")
    #parse inputs
    #read options
    options, arguments = p.parse_args()
    a = options.algo 
    #print a 
    path = '/afs/cern.ch/user/s/sganju/private/'

    #know all of 2014 
    filename_2014 = glob.glob(path + 'dataframe-2014*.csv')
        
    data = pd.read_csv(options.train_dataset)
     
    for each_file in filename_2014:
        week = pd.read_csv(each_file)
	transform_csv() 
        algo(a)
        merge_csv() 

#main ends here 
if __name__ == '__main__':
	proc = Process(target = monitor)
	proc.daemon = True 
	proc.start()
        time0 = time.time()
	main()
	print time.time() - time0 , ' : Total Time for execution'
        print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
