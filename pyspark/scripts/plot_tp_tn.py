import matplotlib.pyplot as plt
import csv 
import pandas as pd 
import numpy as np 
import datetime 

actual = pd.read_csv("actual.csv")
predicted = pd.read_csv("predicted_rf.csv")

actual_tp = actual['TP']
predicted_tp = predicted['TP']

actual_tn = actual['TN']
predicted_tn = predicted['TN']

#subtract values for normalization

tn = pd.DataFrame()
tp = pd.DataFrame()

for each in actual:
	tp = actual_tp - predicted_tp 
	tn = actual_tn - predicted_tn

week = np.ones(52,int)

for each in range(0,51): 
	week[each+1] = week[each] + 1

tp = tp.as_matrix() 

plt.plot(week,tp, label='TRUE POSITIVE')

plt.xlabel('WEEK')
plt.ylabel('TRUE POSITIVE')

plt.title('TRUE POSITIVE 2014 RF')
plt.legend()
plt.show()

plt.plot(week,tn, label='TRUE NEGATIVE')

plt.xlabel('WEEK')
plt.ylabel('TRUE NEGATIVE')

plt.title('TRUE NEGATIVE 2014 RF')
plt.legend()
plt.show()


