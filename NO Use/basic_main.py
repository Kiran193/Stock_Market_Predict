
import datetime
import os
import csv
import pandas as pd
import numpy as np

import numpy as np
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale
from TFANN import ANNR
# from google.colab import files

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
        csv_DF = pd.read_csv(csv_file, delimiter=",", skiprows=1, usecols=(1,4))
        # print(csv_DF.head())
        return csv_DF
    else:
        print ("Can't fetch the file: ",file_name)
        return csv_DF

def clean_train_predict(df):

    
        #gets the price and dates from the matrix
        prices = df[:, 1].reshape(-1, 1)
        dates = df[:, 0].reshape(-1, 1)
        #creates a plot of the data and then displays it
        mpl.plot(dates[:, 0], prices[:, 0])
        mpl.show()


        #Number of neurons in the input, output, and hidden layers
        input = 1
        output = 1
        hidden = 50
        input2 = 1
        output2 = 1
        hidden2 = 50
        #array of layers, 3 hidden and 1 output, along with the tanh activation function 
        layers = [('F', hidden), ('AF', 'tanh'), ('F', hidden), ('AF', 'tanh'), ('F', hidden), ('AF', 'tanh'), ('F', output)]
        layers2 = [('F', hidden2), ('AF', 'tanh'), ('F', hidden2), ('AF', 'tanh'), ('F', hidden2), ('AF', 'tanh'), ('F', output2)]

        #construct the model and dictate params
        mlpr = ANNR([input], layers, batchSize = 256, maxIter = 20000, tol = 0.2, reg = 1e-4, verbose = True) 
        mlpr2 = ANNR([input2], layers2, batchSize = 256, maxIter = 10000, tol = 0.1, reg = 1e-4, verbose = True) 

        ### Training the Model

        #number of days for the hold-out period used to access progress
        holdDays = 5
        totalDays = len(dates)
        #fit the model to the data "Learning"
        mlpr.fit(dates[0:(totalDays-holdDays)], prices[0:(totalDays-holdDays)]) 
        mlpr2.fit(dates[0:(totalDays-holdDays)], prices[0:(totalDays-holdDays)])

        #Predict the stock price using the model
        pricePredict = mlpr.predict(dates)
        pricePredict2 = mlpr2.predict(dates)
        #Display the predicted reuslts agains the actual data
        mpl.plot(dates, prices)
        mpl.plot(dates, pricePredict, c='#5aa9ab')
        mpl.plot(dates, pricePredict2, c='#8B008B')
        mpl.show()


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

        # data=A_df.drop(['Date'],axis=1)
        # A_df["Date"]= A_df["Date"].str.split(",", n = 1, expand = True)
        print(A_df.head())
        print(A_df.isnull().sum()) #no missing values


        #scales the data to smaller values
        A_df=scale(A_df)

        clean_train_predict(A_df)

        

    else:
        print("Empty A.csv...")




if __name__== "__main__":
	main()