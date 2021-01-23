import datetime
import os
import csv
import pandas as pd
import numpy as np

import numpy as np
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale
from TFANN import ANNR


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
        csv_DF = pd.read_csv(csv_file)

        csv_DF["Date"] = pd.to_datetime(csv_DF["Date"])
        csv_DF["Date"] = csv_DF["Date"].dt.strftime("%Y-%m-%d")

        print(csv_DF.head())
        return csv_DF
    else:
        print ("Can't fetch the file: ",file_name)
        return csv_DF

def insert_stock(stock_performance, conn, cur):
        """ Insert Performance data into the DB. """

        stock_performance["bse_code"].fillna(-1, inplace=True)
        stock_performance = stock_performance.astype({"bse_code": int})
        stock_performance = stock_performance.astype({"bse_code": str})
        stock_performance["bse_code"] = stock_performance["bse_code"].replace(
            '-1', np.nan)

        stock_performance = stock_performance[['company_code', '1day', '5day', '30day', '90day', '6month', '1year', '2year', '5year', 'date', 'nse_code', 'bse_code']]

        exportfilename = "stock_performance.csv"
        exportfile = open(exportfilename, "w")
        stock_performance.to_csv(
            exportfile, header=True, index=False, float_format="%.2f", line_terminator='\r')
        exportfile.close()

        copy_sql = """
            COPY dash_process.stock_performance FROM stdin WITH CSV HEADER
            DELIMITER as ','
            """

        with open(exportfilename, 'r') as f:
            cur.copy_expert(sql=copy_sql, file=f)
            conn.commit()
        f.close()
        os.remove(exportfilename)



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
        # print(A_df.head())
        # print(A_df.isnull().sum()) #no missing values


        #scales the data to smaller values
        # A_df=scale(A_df)

        # clean_train_predict(A_df)

    else:
        print("Empty A.csv...")


if __name__== "__main__":
	main()