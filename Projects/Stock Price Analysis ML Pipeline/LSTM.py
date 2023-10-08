#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM: In the below we create a LSTM model that 
will be trained on historic stock data.  Our goal
is to train the model so that we can predict what the
close stock of a price will be based on the date and
the opening price of the stock.

Several functions are utilized below (more notes can
be found about the functions within their declaration):
    
    ss - Function to standardize data
    
    mm - Function to normalize data
    
    umm - Function to denormalize data
    
    lstm_model - the LSTM model
    
The Architecture of the LSTM model is in the
LSTM_ARCH.py file.  For this model, we use only one 
hidden layer with 16 neurons.  

PySpark is used so that we can read data from our
historical MySQL table and so that we can write data 
to our prediction MySQL table.

Comments have been input throughout the below to
help document.
"""
# Import necessary libraries
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from LSTM_ARCH import LSTM
from sklearn.model_selection import train_test_split

def ss(X):
    """
    Standardize the data by using the following formula:
        
        z = x - u / o
        
    where x is the scalar value of the data being standardized,
    u is the mean, and o is the standard deviation
    """
    X = X.values # get all values as a matrix
    sc = StandardScaler() # Standard Scaler object
    xsc = sc.fit_transform(X) # Standardize the data
    return xsc

def mm(X):
    """
    Normalize the data using min max normalization
    
    x = x - min(x) / max(x) - min(x)
    """
    X = X.values # get all values as a matrix
    mm = MinMaxScaler() # MinMax Scaler object
    xmm = mm.fit_transform(X.reshape(-1,1)) # Normalize the data
    return xmm

def umm(X, minn, maxx):
    """
    Un-Normalize the data using the below formula
    
    x = x - min(x) / max(x) - min(x)
    
    x = (x * (max(x) - min(x))) + min(x)
    """
    X = (X * (maxx - minn)) + minn
    return X

def lstm_model(X_train, y_train, X_test, y_test, epochs):
    """
    Use the LSTM model defined in the LSTM_ARCH.py file.
    """
    # Initialize the model
    model = LSTM()
    # Use Adaptive Moment Estimation (ADAM) as the optimizer, use weight decay
    # to prevent overfitting
    optimizer = optim.Adam(model.parameters(), weight_decay=.00001)
    # Use MSE as our loss
    loss_fn = nn.MSELoss()
    # Load our data 
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False)
    # Train and test the model
    n_epochs = epochs
    for epoch in range(n_epochs):
        # Train the model
        model.train()
        for X_batch, y_batch in loader:
            # Compute model prediction
            y_pred = model(X_batch)
            # Compute the loss using MSE
            loss = loss_fn(y_pred, y_batch)
            # Call the ADAM optimizer
            optimizer.zero_grad()
            # Back propagate the error
            loss.backward()
            # Update weights
            optimizer.step()
        # Test the model
        model.eval()
        with torch.no_grad():
            # Compute model prediction
            y_pred = model(X_train)
            # Compute the MSE for training data
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            # Compute model prediction
            y_pred = model(X_test)
            # Compute the MSE for the test data
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

    return y_test, y_pred, model.eval()

def main():

    # Create a Spark Session using SparkSession.builder - config set to current jar version
    spark = SparkSession.builder \
        .master("local[1]") \
        .config("spark.jars", "mysql-connector-j-8.1.0.jar") \
        .appName("TEST") \
        .getOrCreate() 
    
    # Read from MySQL Table - 
    # This table holds the historical stock data that we obtained from
    # the API call in the ETL.py file
    pysparkDFH = spark.read \
        .format("jdbc") \
        .option("driver","com.mysql.cj.jdbc.Driver") \
        .option("url", "jdbc:mysql://localhost:####/STOCKPRD") \
        .option("dbtable", "HISTORICAL_DATA") \
        .option("user", "##########") \
        .option("password", "##########") \
        .load()
        
    # Read from MySQL Table -
    # This table is for our prediction data
    pysparkDFP = spark.read \
        .format("jdbc") \
        .option("driver","com.mysql.cj.jdbc.Driver") \
        .option("url", "jdbc:mysql://localhost:####/STOCKPRD") \
        .option("dbtable", "PREDICTION_DATA") \
        .option("user", "##########") \
        .option("password", "##########") \
        .load()
        
    # Create a pandas dataframe from the pyspark dataframe  
    # One for our historical data and one for our prediction
    pandasDFH = pysparkDFH.toPandas()
    pandasDFP = pysparkDFP.toPandas()
    
    # Our LSTM model will only use the date (numeric) and the open 
    # price of the stock as input
    X = pandasDFH[['date_numeric', 'open']]
    # Set y to our target value (the closing price of the stock)
    y = pandasDFH['close']
    
    # Use scikit learn to split the data - note that we want shuffle to be false
    # since we are dealing with time series data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)
    
    # Get the min and max values of our test set y variables (closing price of stock)
    # We will need these values to "de-normalize" our data
    minn = min(y_test)
    maxx = max(y_test)
    
    # Standardize X, Normalize y
    X_train = ss(X_train)
    X_test = ss(X_test)
    y_train = mm(y_train)
    y_test = mm(y_test)
    
    # Print data shape (for testing/checks)
    print("Training Shape:", X_train.shape, y_train.shape)
    print("Testing Shape:", X_test.shape, y_test.shape) 
    
    # Convert data to pytorch tensors
    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors = Variable(torch.Tensor(X_test))
    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))
    
    # Create and call lstm model
    # We'll train the model for 100 epochs
    epochs = 100
    yt, yp, m = lstm_model(X_train_tensors, y_train_tensors, X_test_tensors, y_test_tensors, epochs)
    
    # Convert pytorch tensors to numpy arrays for de-normalization
    yp = yp.detach().cpu().numpy()
    yt = yt.detach().cpu().numpy()
    
    # De-normalize the data
    ypmod = umm(yp, minn, maxx)
    ytmod = umm(yt, minn, maxx)
    
    # Plot our model's prediction vs the true results
    plt.plot(ytmod, label = "True Values")
    plt.plot(ypmod, label = "Predicted Values")
    plt.title("Stock Analysis")
    plt.legend()
    plt.show()
    
    # For predicting tomorrow's closing price we are going to make
    # a naive assumption and assume that the stock price will
    # have the same opening price that it did for today
    current_day_data = pandasDFH.tail(1)
    
    # Reset/drop index
    current_day_data = current_day_data.reset_index(drop = True)
    
    # Join the current days stock data with the data we have
    # in our prediction_data table
    # Note that we only want the 'open' price added in
    data = pandasDFP.join(current_day_data[['open']])
    data = data[['date_numeric', 'open']]
    
    # Standardize the data for the LSTM model
    data = ss(data)
    
    # Convert to torch tensor
    data = Variable(torch.Tensor(data))
    
    # Pass tomorrow's date with the stock open price
    # to our model to try and predict what the closing
    # price of the stock will be tomorrow
    m.eval()
    with torch.no_grad():
        # Compute model prediction
        y_pred = m(data)
        # De-normalize the data
        ypmod = umm(y_pred, minn, maxx)
    
    # Add our prediction to our dataframe
    pandasDFP["prediction"] = ypmod
    
    # Convert the pandas df to a spark df so that we can input
    # the results into our prediction_data table in MySQL
    sparkFDF = spark.createDataFrame(pandasDFP)
    
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
      
    # Print prediction
    print(pandasDFP)
    
main()
