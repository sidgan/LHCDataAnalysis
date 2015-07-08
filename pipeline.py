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



def cal_score(method, clf, features_test, target_test):
		scores = cross_val_score(clf, features_test, target_test)
		print method + " : %f " % scores.max()
		#print scores.max()		

def main():
	type_of_problem = ""
	split = 0.3
	su_train = []
	su_test = [] 
	p = optparse.OptionParser()
	#take path of training data set 
	p.add_option('--path_train', '-p', default='/afs/cern.ch/user/s/sganju/private/2014_target.csv')
	#what type of problem is it? regression/classification/clustering/dimensionality reduction
	p.add_option('--type_of_problem', '-t', default = 'c')
	#include cross validation true/false
	p.add_option('--cross_validation', '-v', default ='True')
	#take the numerical values 
	#p.add_option('--numerical_values', '-n')
	#specify target column
	p.add_option('--target', '-y')	
	options, arguments = p.parse_args()

	num_values = "id cpu creator dataset dbs dtype era naccess nblk	nevt nfiles nlumis nrel nsites nusers parent primds proc_evts procds rel1_0 rel1_1 rel1_2 rel1_3 rel1_4	rel1_5 rel1_6 rel1_7 rel2_0 rel2_1 rel2_10 rel2_11 rel2_2 rel2_3 rel2_4 rel2_5 rel2_6 rel2_7 rel2_8 rel2_9 rel3_0 rel3_1 rel3_10 rel3_11 rel3_12 rel3_13 rel3_14 rel3_15 rel3_16 rel3_17 rel3_18 rel3_19 rel3_2 rel3_20 rel3_21 rel3_22 rel3_23 rel3_24 rel3_25 rel3_26 rel3_3 rel3_4 rel3_5 rel3_6 rel3_7 rel3_8 rel3_9 relt_0 relt_1 relt_2 rnaccess rnusers rtotcpu s_0 s_1 s_2 s_3 s_4size tier totcpu wct"
	num_values = num_values.split()
	
	#load from files 
	train = pd.read_csv(options.path_train)
		
	#load target values 
	target = train['target']
	
	#TRAINING DATA SET 
	#final data frame with categorical and numerical values 
	#data = pd.concat([train.get(num_values), su_train], axis=1)
        data = train 
	#data = data.head(400)
	#target = target.head(400)
	
	
	print "Performing imputation."
	imp = data.dropna().mean()
	test = data.fillna(imp)
        data = data.fillna(imp)

	print "Splitting the training data with %f." % split 
	features_train, features_test, target_train, target_test = train_test_split(data, target, test_size=split, random_state=0)
	print "Generating Model"	
	#diffrentiate on the basis of type of problem
	#RANDOM FOREST CLASSIFIER 
	rf = RandomForestClassifier(n_estimators=100)
	rf = rf.fit(features_train, target_train)
	cal_score("RANDOM FOREST CLASSIFIER",rf, features_test, target_test)

	#Ada boost 
	clf_ada = AdaBoostClassifier(n_estimators=100)
	params = {
		'learning_rate': [.05, .1,.2,.3,2,3, 5],
		'max_features': [.25,.50,.75,1],
		'max_depth': [3,4,5],
		}
	gs = GridSearchCV(clf_ada, params, cv=5, scoring ='accuracy', n_jobs=4)
	clf_ada.fit(features_train, target_train)
	cal_score("ADABOOST",clf_ada, features_test, target_test)

	#RANDOM FOREST CLASSIFIER 
	rf = RandomForestClassifier(n_estimators=100)
	rf = rf.fit(features_train, target_train)
	cal_score("RANDOM FOREST CLASSIFIER",rf, features_test, target_test)
	#predictions = rf.predict_proba(test)
	#Gradient Boosting
	gb = GradientBoostingClassifier(n_estimators=100, subsample=.8)
	params = {
		'learning_rate': [.05, .1,.2,.3,2,3, 5],
		'max_features': [.25,.50,.75,1],
		'max_depth': [3,4,5],
	}
	gs = GridSearchCV(gb, params, cv=5, scoring ='accuracy', n_jobs=4)
	gs.fit(features_train, target_train)
	cal_score("GRADIENT BOOSTING",gs, features_test, target_test)
	#sorted(gs.grid_scores_, key = lambda x: x.mean_validation_score)
	#print gs.best_score_
	#print gs.best_params_
	#predictions = gs.predict_proba(test)
	#KERNEL APPROXIMATIONS - RBF 		
	rbf_feature = RBFSampler(gamma=1, random_state=1)
	X_features = rbf_feature.fit_transform(data)
	
	#SGD CLASSIFIER		
	clf = SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
    		fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
      loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
       random_state=None, shuffle=True, verbose=0,
       warm_start=False)
	clf.fit(features_train, target_train)
	cal_score("SGD Regression",clf, features_test, target_test)


	#KN Classifier
	neigh = KNeighborsClassifier(n_neighbors = 1)
	neigh.fit(features_train, target_train)
	cal_score("KN CLASSIFICATION",neigh, features_test, target_test)
	#predictions = neigh.predict_proba(test)
	
		

	
	#Decision Tree classifier
	clf_tree = tree.DecisionTreeClassifier(max_depth=10)
	clf_tree.fit(features_train, target_train)
	cal_score("DECISION TREE CLASSIFIER",clf_tree, features_test, target_test)
	#predictions = clf_tree.predict_proba(test)
	
	

#main ends here 
if __name__ == '__main__':
	main()
