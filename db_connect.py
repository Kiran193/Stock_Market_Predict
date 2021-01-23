import csv
import pandas as pd
from sqlalchemy import create_engine, types
import datetime
import os

engine = create_engine('mysql://root:root@localhost/db_stock_market') # enter your password and database names here


def read_csv_file(csv_file):
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file,sep=',',quotechar='\'',encoding='utf8') # Replace Excel_file_name with your excel sheet name
        return df
    else:
        return "Not Present...!!!"
    return df

def data_clean(df):
    
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    return df


def store_table(df, table_name, engine):
    
    df.to_sql(table_name,con=engine,index=False,if_exists='append') # Replace Table_name with your sql table name

    return "Stored in " + table_name

def main():

    


