import numpy as np
from datetime import datetime
import smtplib
import time
import datetime

import pandas as pd
import os
from selenium import webdriver
#For Prediction
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing,  svm, model_selection #cross_validation,

my_path = os.path.abspath(os.getcwd())
# print("...."+ my_path)
# filepath = os.path.join(my_path,"CSV_Files\\")

# print("file_name..." + file_name)
# csv_file = filepath + "A.csv"



def prediction(csv_file):
    #Outputting the Historical data into a .csv for later use
    df = pd.read_csv(csv_file, delimiter=",",  usecols=['Date','Open','High','Low','Close','Adj Close'])

    df["Date"] = pd.to_datetime(df.Date)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    df = df.sort_values('Date')
    # print(df.isnull().sum()) #no missing values
    df.describe()
    df.drop_duplicates()

    df['prediction'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    # print(df.head())
    # print(df.column())

    forecast_time = int(5)

    df = df.drop(["Date"],axis=1)
    X = np.array(df.drop(['prediction'], 1))
    Y = np.array(df['prediction'])
    X = preprocessing.scale(X)
    X_prediction = X[-forecast_time:]
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.5)

    #Performing the Regression on the training data
    clf = LinearRegression()
    clf.fit(X_train, Y_train)
    prediction = (clf.predict(X_prediction))

    # print(prediction)

    stock = csv_file.split(".",1)
    stock_name = stock[0]
    # print("Stock Name..." + stock_name)

    #If the predicted price of the stock is at least 1 greater than the previous closing price
    last_row = df.tail(1)

    output = ''
    if (float(prediction[4]) > (float(last_row['Close']))):
        output = ("\n\nStock:" + str(stock_name) + 
                  "\nPrior Close:\n" + str(last_row['Close']) + 
                  "\n\nPrediction in 1 Day: " + str(prediction[0]) + 
                  "\nPrediction in 5 Days: " + str(prediction[4]))

    # print("Final ......\n")
    # print(output)
    return output

def main():

    A = "A.csv"
    ABB = "ABB.csv"
    ABM = "ABM.csv"
    ABUS = "ABUS.csv"
    data_csv = "data_csv.csv"

    # Sort DataFrame by date
    # A_df = A_df.sort_values('Date')
    # print(A_df.isnull().sum()) #no missing values

    A_df = prediction(A)
    ABB_df = prediction(ABB)
    ABM_df = prediction(ABM)
    ABUS_df = prediction(ABUS)

    print(A_df)
    print(ABB_df)
    print(ABM_df)
    print(ABUS_df)



if __name__== "__main__":
	main()