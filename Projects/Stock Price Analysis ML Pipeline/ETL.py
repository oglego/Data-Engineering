#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In the below python script we will create an ETL job for 
Stock Price Analysis.

We are interested in analyzing the stock prices for Ford
(Ford Motor Company).  To obtain historical stock data we 
will utilize the Polygon.IO API.  For the current days stock
price we will use yfinance.

Note that we can analyze any stock that we want, we just need to
update the ticker variable below.

With the API call, a JSON will be returned that holds the stock
data for Ford.  We will parse through this JSON file and add the 
values wanted to a data dictionary.  

The values we are interested in are:
    
    c - stock close price
    
    h - highest price of the stock during the day 
    
    l - lowest price of the stock during the day
    
    n - number of transactions
    
    o - opening price of the stock
    
    t - unix timestamp for the stock
    
    v - trading volume
    
Note that further information on these attributes can be found
in the documentation for Polygon.IO.

Once we have populated a dictionary with the data, we will then
convert it into a pandas dataframe.  After some cleaning/modifications
of the dataframe, we will then utilize PySpark so that we can insert
the data from the API call into a table that we have created in MySQL.

Ultimately, our goal is to act as a day trader and be able to predict
what the stock price will be at close based on what the stock price is
at open.  We will use historical data to train a LSTM model so that we
can try and predict what the stock price will be at close.
"""

# Import necessary libraries
import requests
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd
from pyspark.sql import SparkSession

# Through further investigation, the Polygon.IO API does not return current
# day stock data - for current day stock data we will use yfinance - yfinance
# allows us to download market data from Yahoo! Finance's API 
import yfinance as yf

# Create a variable to hold the ticker for the stock that we are interested in
# note that in the below we use F for Ford Motor Company
ticker = "F"

# Create two variables to store dates that will be passed
# as parameters to the API calls url string
current_date = str(date.today())
start_date = str((datetime.now() - relativedelta(years=2)).strftime('%Y-%m-%d'))

# Create a list that will hold the current date - note that we have this because
# it can be modified if we want to look further into the future with multiple
# days
future_dates = []
for i in range(1):
    future_dates.append(str((datetime.now() + relativedelta(days=i)).strftime('%Y-%m-%d')))

# Before our API calls, we are going to create a pandas dataframe for the above date(s)
# We use "date(s)" since the above statement can be modified to allow more than one date
fdf = pd.DataFrame(future_dates)

# Rename column to date instead of 0
fdf = fdf.rename(columns={0:"date"})

# Create a column to store a numeric version of the date
fdf["date_numeric"] = pd.to_datetime(fdf["date"]).dt.strftime("%Y%m%d").astype(int)

# Create a prediction column for the stock price and set it to null
fdf["prediction"] = "NULL"

# Add a column for the stock ticker
fdf["ticker"] = ticker

# --------------------------
# API call to get stock data
# --------------------------

# Note in the URL we pass the ticker parameter/variable as well as the dates we are interested in
response = requests.get("https://api.polygon.io/v2/aggs/ticker/"+ticker+"/range/1/day/"+start_date+"/"+current_date+"?adjusted=true&sort=asc&&apiKey=APIKEY")

# Set response equal to JSON values
response = response.json()

# Create a dictionary for the data obtained in the JSON
data = {}

# Iterate through the JSON obtained from the API call and
# add the data into our data dictionary
for val in response["results"]:
    # Convert the unix numeric timestamp to a date - the date is our key and the other
    # attributes are the values for our dictionary
    data[datetime.utcfromtimestamp(val['t'] / 1000.0).strftime('%Y-%m-%d')] = val['o'], val['h'], val['l'], \
                                                                              val['c'], val['n'], val['v'], val['t']

# Convert the dictionary into a pandas dataframe                                                                             
df = pd.DataFrame.from_dict(data, orient = 'index')

# Give the columns in the dataframe names
df = df.rename(columns={0:"open", 1:"high", 2:"low", 3:"close", \
                        4:"num_of_trans", 5:"trading_volume", 6:"unix_timestamp"})

# Add a column for the stock ticker 
df["ticker"] = ticker

# Our index is currently the date, we are going to reset this.
df = df.reset_index()

# Rename the reset column as date since it now houses the date
df = df.rename(columns={"index":"date"})

# Create a numeric version of the date
df["date_numeric"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d").astype(int)

# Since we have a numeric version of the date, we're going to drop the date column
df = df.drop(["date"], axis=1)

# We need to add todays stock data to our df

# The below steps use yfinance to get todays stock data, we then clean the
# results so that they line up more with our historical data that we
# obtained with our API call
curr = yf.Ticker(ticker)
curr = curr.history(period="1day")
curr = curr.drop(['Dividends', 'Stock Splits', 'Volume'], axis=1)
curr = curr.reset_index()
curr["num_of_trans"] = 0
curr["trading_volume"] = 0
curr["unix_timestamp"] = 0
curr = curr.rename(columns={"Open":"open", "High":"high", "Low":"low", "Close":"close", "Date":"date"})
curr["ticker"] = ticker
curr["date_numeric"] = pd.to_datetime(curr["date"]).dt.strftime("%Y%m%d").astype(int)
curr = curr.drop(["date"], axis=1)

# Now we need to add the current days stock data to our historical data
df = df.append(curr, ignore_index=True)

# Create a Spark Session using SparkSession.builder - config set to current jar version
spark = SparkSession.builder \
    .master("local[1]") \
    .config("spark.jars", "mysql-connector-j-8.1.0.jar") \
    .appName("STOCKETL") \
    .getOrCreate() 
    
# Create a spark dataframe from our pandas dataframe

# We will do this for both the historical data and our prediction data
sparkDF = spark.createDataFrame(df)
sparkFDF = spark.createDataFrame(fdf)

# Write the data into our historical_data and prediction_data tables in MySQL

# We use overwrite in case there is anything currently in the table

# HISTORICAL DATA WRITE
# ---------------------
sparkDF.write \
  .format("jdbc") \
  .mode("overwrite") \
  .option("driver","com.mysql.cj.jdbc.Driver") \
  .option("url", "jdbc:mysql://localhost:####/STOCKPRD") \
  .option("dbtable", "HISTORICAL_DATA") \
  .option("user", "##########") \
  .option("password", "##########") \
  .save()
  
# PREDICTED DATA WRITE
# --------------------
sparkFDF.write \
  .format("jdbc") \
  .mode("overwrite") \
  .option("driver","com.mysql.cj.jdbc.Driver") \
  .option("url", "jdbc:mysql://localhost:####/STOCKPRD") \
  .option("dbtable", "PREDICTION_DATA") \
  .option("user", "##########") \
  .option("password", "##########") \
  .save()




