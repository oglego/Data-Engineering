#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
from config import config
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from producer import producer
from consumer import consumer

def api_call(key, start_date, current_date, crypto) -> dict:
    """
    Make an API call to the specified endpoint using the requests library
    
    Note that all of the parameters are instantiated in the config file
    
    Parameters:
        key (str): API Key 
        start_date (str): Starting date to pass to the API url string
        current_date (str): Todays date - acts as the end date to pass to API url string
        crypto (str): Crypto to pass to the API url string
    
    Returns:
        dict: The JSON response from the API.
    
    """
    response = requests.get("https://api.polygon.io/v2/aggs/ticker/"+crypto+"/range/1/day/"+start_date+"/"+current_date+"?adjusted=true&sort=asc&&apiKey="+key)
    response_json = response.json()
    return response_json

def cumulative_moving_average(data) -> list:
    """
    Calculate the cumulative moving averages for a given list of numerical data.
    
    The cumulative moving average at each index is computed as the average of all data 
    points from the beginning up to the current index.
    
    Parameters:
        data (list): A list of numerical data to calculate cumulative moving averages for.
    
    Returns:
        list: A list of cumulative moving averages for the input data.
    """
    # Convert the list of prices into a pandas series
    series = pd.Series(data)
    # Get the window of series of prices till current
    windows = series.expanding()
    # Create a series of moving averages 
    moving_averages = windows.mean()
    # Convert pandas series back to list
    moving_averages = moving_averages.tolist()
    return moving_averages

def _plot(prices, moving_average):
    """
    Plot the given prices and their corresponding moving average.
    
    This function plots the prices and their moving average on a graph to visualize
    the relationship between the price changes and the moving average.
    
    Parameters:
        prices (list): A list of numerical prices to be plotted on the y-axis.
        moving_average (list): A list of moving average values to be plotted on the y-axis.
    
    Returns:
        None
    """
    plt.plot(prices)
    plt.plot(moving_average)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Ethereum Moving Average")
    
def main():
    """
    Use the config file to set values for the key, start_date, current_date,
    and crypto parameters.
    """
    key = config["key"]
    start_date = config["start_date"]
    current_date = config["current_date"]
    crypto = config["crypto"]
    
    # Store the results of the JSON file that is returned from the API call
    raw_data = api_call(key, start_date, current_date, crypto)
    eth_data = raw_data["results"]
    
    # Send Ethereum data to the Kafka Producer
    producer.send('ETH', eth_data)
    # Instantiate a Kafka Consumer for reading
    eth_consumer = consumer
    
    # Iterate through the messages received from the consumer
    for message in eth_consumer:
        # Extract message
        msg = message.value
        # Close Consumer after processing one message
        eth_consumer.close()

    # Create a dictionary for the data obtained in the JSON
    eth_dict = {}

    # Update our dictionary based on the message
    for val in msg:
        # Convert the unix numeric timestamp to a date - the date is our key and the other
        # attributes are the values for our dictionary
        eth_dict[datetime.utcfromtimestamp(val['t'] / 1000.0).strftime('%Y-%m-%d')] = val['c']
        
    # Ethereum prices that were returned
    prices = eth_dict.values()
    # Moving average of Ethereum prices
    moving_average = cumulative_moving_average(prices)
    # Plot Price with Moving Average
    _plot(prices, moving_average)
    
main()
    

