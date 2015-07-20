#!usr/bin/env python 
#author sidgan

from sklearn.kernel_approximation import RBFSampler
import sklearn.cluster 
import optparse
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation 
from sklearn.metrics import precision_recall_curve
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm 
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
import random
import csv as csv
import pandas as pd 
import numpy as np 
import warnings 
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', SyntaxWarning)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 
import matplotlib.pyplot as plt 
import sklearn
from sklearn.ensemble import AdaBoostClassifier


#accuracy 
def cal_score_accuracy(method, clf, features_test, target_test):
		scores = cross_val_score(clf, features_test, target_test)
		print method + " : %f " % scores.max()
		#print scores.max()		


#recall 
def cal_score_recall(method, clf, features_test, target_test):
		scores = cross_val_score(clf, features_test, target_test)
		y_true = np.array([])
		y_scores = np.array([])
		precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
		print method + " : %f precision" % precision
		print method + " : %f recall" % recall
		print method + " : %f thresholds" % thresholds
		#print scores.max()		



#F1 
def cal_score_f1(method, clf, features_test, target_test):
		scores = cross_val_score(clf, features_test, target_test)
		print method + " : %f " % scores.max()
		#print scores.max()		





def main():
	split = 0.3
	num_values = "id cpu creator dataset dbs dtype era naccess nblk	nevt nfiles nlumis nrel nsites nusers parent primds proc_evts procds rnaccess rnusers rtotcpu size tier totcpu wct"
	p = optparse.OptionParser()
	#take training data set 
	p.add_option('--train_dataset', '-i', default='/afs/cern.ch/user/s/sganju/private/2014_target.csv')
	#specify target column
	p.add_option('--target', '-y', default="target")	
	p.add_option('--num_values', '-o', default= num_values)
	#parse inputs
	options, arguments = p.parse_args()
	#split different numerical values
	num_values = num_values.split()
	
	#load from files 
	train = pd.read_csv(options.train_dataset)
		
	#load target values 
	target = train['target']
	
	#TRAINING DATA SET 
	#final data frame with categorical and numerical values 
	#data = pd.DataFrame()
	#data  train.(num_values)
	#data = pd.concat([train.get(num_values)], axis=1)
	data = train 
	#data = data.head(400)
	#target = target.head(400)
	
	#no NAN values so imputation not necessary
	#print "Performing imputation."
	#imp = data.dropna().mean()
	#test = data.fillna(imp)
    #data = data.fillna(imp)

	print "Splitting the training data with %f." % split 
	features_train, features_test, target_train, target_test = train_test_split(data, target, test_size=split, random_state=0)
	print "Generating Model"	
	#diffrentiate on the basis of type of problem
	#RANDOM FOREST CLASSIFIER 
	rf = RandomForestClassifier(n_estimators=100)
	rf = rf.fit(features_train, target_train)
	cal_score_accuracy("RANDOM FOREST CLASSIFIER",rf, features_test, target_test)
	#test data set then make predictions 
	test = pd.read_csv('dataframe-20130101-20130107.csv')
	#test = pd.concat([train.get(num_values)], axis=1)
	predictions = rf.predict_proba(test)
	#predict for a week and then print it 
	print predictions
	

#main ends here 
if __name__ == '__main__':
	main()
