#### Stock Market Predictions with LSTM in Python 
#### From DataCamp Author: Thushan Ganegendara

# import datetime
# import os
# import csv
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as mpl
# from sklearn.preprocessing import scale
# from TFANN import ANNR

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler

# my_path = os.path.abspath(os.getcwd())
# # print("...."+ my_path)
# filepath = os.path.join(my_path,"CSV_Files\\")

def read_csv_file(file_name):

    # print("file_name..." + file_name)
    # csv_file = filepath + file_name
    csv_file = file_name

    # print("CSV Path..." + csv_file)

    csv_DF = pd.DataFrame()

    if os.path.isfile(csv_file):
        csv_DF = pd.read_csv(csv_file, delimiter=",",  usecols=['Date','Open','High','Low','Close','Adj Close'])
        # print(csv_DF.head())
        return csv_DF
    else:
        print ("Can't fetch the file: ",file_name)
        return csv_DF

def df_visual(df):

    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
    plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.show()

def train_test(df):
    # First calculate the mid prices from the highest and lowest
    high_prices = df.loc[:,'High'].as_matrix()
    low_prices = df.loc[:,'Low'].as_matrix()
    mid_prices = (high_prices+low_prices)/2.0

    print("mid_price.shape()")
    print(mid_prices)

    train_data = mid_prices[:3843]
    test_data = mid_prices[3843:]

    return train_data, test_data

def normalizing(train_data, test_data):
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)

    return scaler, train_data, test_data


def main():

    A = "A.csv"
    ABB = "ABB.csv"
    ABM = "ABM.csv"
    ABUS = "ABUS.csv"
    data_csv = "data_csv.csv"

    print("\n \t\t Read file A.csv")
    A_df = read_csv_file(A)

    if not A_df.empty:
        print("Data Present...")

        A_df["Date"]= A_df["Date"].str.split(",", n = 1, expand = True)
        print("Number of Rows: ", len(A_df.axes[0])) 
        print("Number of col: ", len(A_df.axes[1])) 
        print(A_df.head())

        # Sort DataFrame by date
        A_df = A_df.sort_values('Date')
        print(A_df.isnull().sum()) #no missing values

        # Visual 
        df_visual(A_df)

        # Train, Test
        train_df, test_df = train_test(A_df)

        # Normlizing
        scaler, train_df, test_df = normalizing(train_df, test_df)


        # Train the Scaler with training data and smooth data
        smoothing_window_size = 1281
        for di in range(0,10000,smoothing_window_size):
            
            print(".....................................")
            print(di)

            scaler.fit(train_df[di:di+smoothing_window_size,:])
            print("........Before Train and Smooth ........")

            train_df[di:di+smoothing_window_size,:] = scaler.transform(train_df[di:di+smoothing_window_size,:])

            print("........Train and Smooth ........")


        # You normalize the last bit of remaining data
        scaler.fit(train_df[di+smoothing_window_size:,:])
        train_df[di+smoothing_window_size:,:] = scaler.transform(train_df[di+smoothing_window_size:,:])


        ### Reshape 
        # Reshape both train and test data
        train_data = train_data.reshape(-1)

        # Normalize test data
        test_data = scaler.transform(test_data).reshape(-1)

        ### Smoothing  train Data
        # Now perform exponential moving average smoothing
        # So the data will have a smoother curve than the original ragged data
        EMA = 0.0
        gamma = 0.1
        for ti in range(11000):
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

            if pred_idx >= N:
                date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
            else:
                date = df.loc[pred_idx,'Date']

            std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
            mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
            std_avg_x.append(date)

        print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))


        plt.figure(figsize = (18,9))
        plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
        plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
        #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Mid Price')
        plt.legend(fontsize=18)
        plt.show()






    else:
        print("Empty A.csv...")




if __name__== "__main__":
	main()