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

from sqlalchemy import create_engine, types
from matplotlib import pyplot as plt


my_path = os.path.abspath(os.getcwd())
# print("...."+ my_path)
# filepath = os.path.join(my_path,"CSV_Files\\")

# print("file_name..." + file_name)
# csv_file = filepath + "A.csv"

engine = create_engine('mysql://root:root@localhost/db_stock_market') # enter your password and database names here


def read_csv_file(file_name):

    # print("file_name..." + file_name)
    # csv_file = filepath + file_name
    csv_file = file_name

    stock = csv_file.split(".",1)
    stock_name = stock[0]
    # print("Stock Name..." + stock_name)

    # print("CSV Path..." + csv_file)

    csv_DF = pd.DataFrame()

    if os.path.isfile(csv_file):
        csv_DF = pd.read_csv(csv_file)

        csv_DF["Date"] = pd.to_datetime(csv_DF["Date"])
        csv_DF["Date"] = csv_DF["Date"].dt.strftime("%Y-%m-%d")

        csv_DF = csv_DF.sort_values('Date')
        print(csv_DF.isnull().sum()) #no missing values
        csv_DF.describe()
        csv_DF.drop_duplicates()

        # print(csv_DF.head())
        return csv_DF, stock_name
    else:
        print ("Can't fetch the file: ",file_name)
        return csv_DF, stock_name

def store_table(df,stock_name):

    table_name = "tbl_stock_" + stock_name.lower()    
    df.to_sql(table_name,con=engine,index=False,if_exists='replace') # Replace Table_name with your sql table name

    return "Stored in " + table_name

def LR_prediction(csv_file, df, stock_name):
    # #Outputting the Historical data into a .csv for later use
    # df = pd.read_csv(csv_file, delimiter=",",  usecols=['Date','Open','High','Low','Close','Adj Close'])

    # df["Date"] = pd.to_datetime(df.Date)
    # df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

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
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

    #Performing the Regression on the training data
    clf = LinearRegression()
    model = clf.fit(X_train, Y_train)
    prediction = (clf.predict(X_prediction))

    print("prediction\n")
    print(prediction)

    # print("Score: ", model.score(X_test,Y_test))

    # plot_scatter(Y_test, prediction)
    
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

def plot_scatter(Y_test, prediction):
    plt.scatter(Y_test, prediction)
    plt.xlabel("True Values")
    plt.ylabel("Prediction")

def main():

    A = "A.csv"
    ABB = "ABB.csv"
    ABM = "ABM.csv"
    ABUS = "ABUS.csv"
    data_csv = "data_csv.csv"

    A_df, stock_name = read_csv_file(A)
    print("Hello.....")
    A_df.describe()
    store_table(A_df, stock_name)
    A_df_LR_predict = LR_prediction(A,A_df, stock_name)
    print(A_df_LR_predict)

    ABB_df, stock_name = read_csv_file(ABB)
    store_table(ABB_df, stock_name)
    ABB_df_LR_predict = LR_prediction(ABB,ABB_df, stock_name)
    print(ABB_df_LR_predict)

    ABM_df, stock_name = read_csv_file(ABM)
    store_table(ABM_df, stock_name)
    ABM_df_LR_predict = LR_prediction(ABM,ABM_df, stock_name)
    print(ABM_df_LR_predict)

    ABUS_df, stock_name = read_csv_file(ABUS)
    store_table(ABUS_df, stock_name)
    ABUS_df_LR_predict = LR_prediction(ABUS,ABUS_df, stock_name)
    print(ABUS_df_LR_predict)

    data_csv_df, stock_name = read_csv_file(data_csv)
    store_table(data_csv_df, stock_name)



if __name__== "__main__":
	main()