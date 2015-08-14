#!/bin/bash
#!/usr/bin/env python
#author sidgan 


import matplotlib.pyplot as plt
import csv 
import pandas as pd 
import numpy as np 
import datetime 

"""
File       : plot_cpu.py
Author     : Siddha Ganju <siddhaganju AT gmail dot com>
Description: 

Plot CPU ad RAM usage

"""

actual = pd.read_csv("rf_stats.csv", delimiter = " ")
ram1 = actual['ram']
cpu = actual['cpu']
#active = actual['active']

#divide by total memory 
#value in gigabytes 
#total =  8000000000 
#total bytes 
#total memory used
total = 5376336000  
#from pyutil 
#to normalize, divide by total number of bytes 

interval = 1 

for each in ram1:
	#for normalizing 
	#ram1 = (ram1 / total).astype(float) 
	interval = interval + 1 

interval_x = np.ones(interval-1,int)

for each in range(0,interval-2): 
	interval_x[each+1] = interval_x[each] + 1

plt.plot(interval_x, ram1, label='RAM USAGE')

plt.xlabel('TIME')
plt.ylabel('RAM')

plt.title('RANDOM FOREST RAM USAGE')
plt.legend()
plt.show()

plt.plot(interval_x, cpu,  label='CPU')

plt.xlabel('TIME')
plt.ylabel('CPU')

plt.title('CPU USAGE RANDOM FOREST ')
plt.legend()
plt.show()


