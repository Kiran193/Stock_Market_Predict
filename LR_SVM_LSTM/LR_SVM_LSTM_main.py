##### https://randerson112358.medium.com/predict-stock-prices-using-python-machine-learning-53aa024da20a

#Install the dependencies
import quandl
import numpy as np 
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sqlalchemy import create_engine, types
import matplotlib.pyplot as plt

import LSTM_Predict


my_path = os.path.abspath(os.getcwd())
# print("...."+ my_path)
filepath = os.path.join(my_path,"CSV_Files\\")

# print("file_name..." + file_name)
# csv_file = filepath + "A.csv"

engine = create_engine('mysql://root:root@localhost/db_stock_market') # enter your password and database names here

# Get the stock data
# df = quandl.get("WIKI/AMZN")

def LR_SVM(csv_file):

    df = pd.read_csv(csv_file)

    # Take a look at the data
    # print(df.head())

    # Get the Adjusted Close Price 
    df = df[['Adj Close']] 
    # Take a look at the new data 
    # print(df.head())

    # A variable for predicting 'n' days out into the future
    forecast_out = 7 #'n=7' days
    #Create another column (the target ) shifted 'n' units up
    df['Prediction'] = df[['Adj Close']].shift(-forecast_out)
    #print the new data set
    # print(df.tail())


    ### Create the independent data set (X)  #######
    # Convert the dataframe to a numpy array
    X = np.array(df.drop(['Prediction'],1))

    #Remove the last '7' rows
    X = X[:-forecast_out]
    # print(X)



    ### Create the dependent data set (y)  #####
    # Convert the dataframe to a numpy array 
    y = np.array(df['Prediction'])
    # Get all of the y values except the last '7' rows
    y = y[:-forecast_out]
    # print(y)



    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ################  Models ################

    # Create and train the Support Vector Machine (Regressor) 
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
    svr_rbf.fit(x_train, y_train)


    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    svm_confidence = svr_rbf.score(x_test, y_test)
    print("svm confidence: ", svm_confidence)


    # Create and train the Linear Regression  Model
    lr = LinearRegression()
    # Train the model
    lr.fit(x_train, y_train)

    # Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
    # The best possible score is 1.0
    lr_confidence = lr.score(x_test, y_test)
    print("lr confidence: ", lr_confidence)


    ################  Forecast ################


    # Set x_forecast equal to the last 7 rows of the original data set from Adj Close column
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    print("x_forecast.....")

    print(x_forecast)

    ################  Predict ################


    # Print linear regression model predictions for the next '7' days
    lr_prediction = lr.predict(x_forecast)
    print("LR_Prediction.....")
    print(lr_prediction)



    # Print support vector regressor model predictions for the next '7' days
    svm_prediction = svr_rbf.predict(x_forecast)
    print(svm_prediction)


    plt.plot(lr_prediction, color = 'blue', label = 'LR Predicted Stock Price')
    plt.plot(svm_prediction, color = 'red', label = 'SVM Predicted Stock Price')

    plt.title('Stock Price Prediction')
    plt.xlabel('Time in Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


def store_table(csv_file):

    stock = csv_file.split(".",1)
    stock_name = stock[0]

    table_name = "tbl_stock_" + stock_name.lower() 

    csv_DF = pd.DataFrame()

    if os.path.isfile(csv_file):
        csv_DF = pd.read_csv(csv_file)  

        csv_DF.to_sql(table_name,con=engine,index=False,if_exists='replace') # Replace Table_name with your sql table name

        return "Stored in " + table_name


def main():

    A = filepath + "A.csv"
    ABB = filepath + "ABB.csv"
    ABM = filepath + "ABM.csv"
    ABUS = filepath + "ABUS.csv"
    data_csv = filepath + "data_csv.csv"

    print("Process A.csv")
    store_table(A)
    # LR_SVM(A)
    LSTM_Predict.LSTM_process().LSTM_(A)

    # print("Process ABB.csv")
    # store_table(ABB)
    # LR_SVM(ABB)
    # LSTM_Predict.LSTM_process().LSTM_(ABB)


    # print("Process ABM.csv")
    # store_table(ABM)
    # LR_SVM(ABM)
    # LSTM_Predict.LSTM_process().LSTM_(ABM)

    # print("Process ABUS.csv")
    # store_table(ABUS)
    # LR_SVM(ABUS)
    # LSTM_Predict.LSTM_process().LSTM_(ABUS)




    
if __name__== "__main__":
	main()