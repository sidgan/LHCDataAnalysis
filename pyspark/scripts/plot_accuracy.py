"""
File       : feature_plot.py
Author     : Siddha Ganju <siddhaganju AT gmail dot com>
Description: 

Complete plots 

"""


import math 
import matplotlib.pyplot as plt
import csv 
import pandas as pd 
import numpy as np 
import datetime 

data = pd.read_csv("predicted_sgd.csv", delimiter = ",")

#print rf.head()

week = np.ones(52,int)

tp = data['TP']
fp = data['FP']
tn = data['TN']
fn = data['FN']

tp = tp.as_matrix()
fp = fp.as_matrix()
tn = tn.as_matrix()
fn = fn.as_matrix()

#calculate values 
tpr = np.ones(52, int)
spc = np.ones(52, int)
ppv = np.ones(52, int)
npv = np.ones(52, int)
fpr = np.ones(52, int)
fnr = np.ones(52, int)
fdr = np.ones(52, int)
tnr = np.ones(52,int)

for each in range(0,51): 
        week[each+1] = week[each] + 1
        tpr[each] = tp[each] / (tp[each] + fn[each])
        spc[each] = tn[each] / (tn[each] + fp[each])
        ppv[each] = tp[each] / (tp[each] + fp[each])
        npv[each] = tn[each] / (tn[each] + fn[each])
        fpr[each] = fp[each] / (fp[each] + tn[each])
        fnr[each] = 1 - tpr[each]
        fdr[each] = 1 - ppv[each]
	
#RANDOM FOREST
plt.plot(week, tpr, label='TRUE POSITIVE RATE SGDT ')

plt.xlabel('TIME')
plt.ylabel('SENSITIVITY')

plt.title('TRUE POSITIVE RATE SGD')
plt.legend()
plt.show()

#RANDOM FOREST
plt.plot(week, spc, label='TRUE NEGATIVE RATE SGD ')

plt.xlabel('TIME')
plt.ylabel('SPECIFICITY')

plt.title('TRUE NEGATIVE RATE SGD')
plt.legend()
plt.show()


#RANDOM FOREST
plt.plot(week, ppv, label='POSITIVE PREDICTIVE VALUE SGD ')

plt.xlabel('TIME')
plt.ylabel('PRECISION')

plt.title(' POSITIVE PREDICTIVE VALUE SGD')
plt.legend()
plt.show()

#RANDOM FOREST
plt.plot(week, fpr, label='FALSE POSITIVE RATE SGD ')

plt.xlabel('TIME')
plt.ylabel('FALL-OUT')

plt.title('FALSE POSITIVE RATE SGD')
plt.legend()
plt.show()



#RANDOM FOREST
plt.plot(week, fnr, label='FALSE NEGATIVE RATE SGD ')

plt.xlabel('TIME')
plt.ylabel('FNR')

plt.title('FALSE NEGATIVE RATE SGD')
plt.legend()
plt.show()


#RANDOM FOREST
plt.plot(week, fdr, label=' FALSE DISCOVERY RATE SGD ')

plt.xlabel('TIME')
plt.ylabel('FDR')

plt.title('FALSE DISCOVERY RATE SGD')
plt.legend()
plt.show()


#RANDOM FOREST
plt.plot(week, npv, label='NEGATIVE PREDICTIVE VALUE SGD ')

plt.xlabel('TIME')
plt.ylabel('NPV')

plt.title('NEGATIVE PREDICTIVE VALUE SGD')
plt.legend()
plt.show()




