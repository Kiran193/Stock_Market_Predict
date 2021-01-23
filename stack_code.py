import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
from dateutil.parser import parse
from datetime import datetime, timedelta
from collections import deque

stock_data = []
stock_date = []
stock_value = []
f = open("CSV_Files\\A.csv","r")
data = f.read()
rows = data.split("\n")
rows_noheader = rows[1:len(rows)]

#Separating values from messy `.csv`, putting each value to it's list and also a combined list of both
for row in rows_noheader:
    [date, value] = row[1:len(row)-1].split('\t')
    stock_date.append(date)
    stock_value.append((value))
    stock_data.append((date, value))

#Numpy array of all closing values converted to floats and normalized against the maximum
stock_value = np.array(stock_value, dtype=np.float32)
normvalue = [i/max(stock_value) for i in stock_value]

#Number of closing values and days. Since there is one closing value for each, they both match and there are 4528 of them (each)
nclose_and_days = 0
for i in range(len(stock_data)):
    nclose_and_days+=1

train_data = stock_value[:2264]
test_data = stock_value[2264:]

scaler = MinMaxScaler()

train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

# Train the Scaler with training data and smooth data
smoothing_window_size = 1100
for di in range(0,4400,smoothing_window_size):
    #error occurs here
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# You normalize the last bit of remaining data
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

# Reshape both train and test data
train_data = train_data.reshape(-1)

# Normalize test data
test_data = scaler.transform(test_data).reshape(-1)

# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(1100):
    EMA = gamma*train_data[ti] + (1-gamma)*EMA
    train_data[ti] = EMA

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data,test_data],axis=0)

window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):
    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))