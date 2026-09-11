LSTM-Based Time-Series Forecasting for Stock Price Prediction

Project Overview

This project develops an LSTM (Long Short-Term Memory) based deep learning model for time-series forecasting using historical stock price data.

The project uses historical Apple Inc. (AAPL) stock price data. The model learns patterns from the previous 60 days of closing prices and predicts the closing price for the next day.

Objective

The main objectives of this project are:

Understand time-series data.

Use historical stock prices for forecasting.

Preprocess and normalize the data.

Create sequential data suitable for LSTM.

Develop and train an LSTM neural network.

Predict future stock prices.

Evaluate the model using RMSE and MAE.

Visualize actual and predicted stock prices.

Dataset

Dataset Used

Apple Inc. (AAPL) Historical Stock Price Dataset

The stock data is obtained automatically using the yfinance Python library, so a CSV dataset does not need to be downloaded manually.

Dataset Features

Feature

Description

Date

Date of stock trading

Open

Opening stock price

High

Highest price during the day

Low

Lowest price during the day

Close

Closing stock price

Adj Close

Adjusted closing price

Volume

Number of shares traded

For this project, only the Close price is used for forecasting.

Why LSTM?

LSTM stands for Long Short-Term Memory.

LSTM is a type of Recurrent Neural Network (RNN) designed for sequential and time-series data.

Stock prices are sequential because the order of observations matters. LSTM can learn patterns and dependencies from previous time steps and use them to make predictions.

Methodology

Download AAPL Stock Data
          ↓
Select Closing Price
          ↓
Normalize Data
          ↓
Create 60-Day Sequences
          ↓
Split Data into Training and Testing
          ↓
Reshape Data for LSTM
          ↓
Build LSTM Model
          ↓
Train Model
          ↓
Make Predictions
          ↓
Convert Predictions to Original Scale
          ↓
Evaluate Using RMSE and MAE
          ↓
Visualize Results
          ↓
Predict Next Day's Closing Price

Data Preprocessing

1. Selecting the Close Price

Only the Close column is selected because the objective is to predict the next day's closing price.

2. Normalization

The closing prices are normalized using MinMaxScaler.

The values are converted to a range between 0 and 1. This helps the neural network learn more efficiently.

3. Creating Sequences

The model uses the previous 60 days of closing prices to predict the next day's closing price.

For example:

Day 1 - Day 60  → Predict Day 61
Day 2 - Day 61  → Predict Day 62
Day 3 - Day 62  → Predict Day 63

LSTM Model Architecture

Input
  ↓
LSTM (50 units)
  ↓
Dropout (20%)
  ↓
LSTM (50 units)
  ↓
Dropout (20%)
  ↓
Dense (25 units)
  ↓
Dense (1 unit)
  ↓
Output

Model Details

Layer

Configuration

Purpose

LSTM

50 units

Learn time-series patterns

Dropout

0.2

Reduce overfitting

LSTM

50 units

Learn additional sequential patterns

Dropout

0.2

Reduce overfitting

Dense

25 units

Process learned features

Dense

1 unit

Generate final prediction

Model Configuration

Optimizer: Adam

Loss Function: Mean Squared Error

Epochs: 20

Batch Size: 32

Training Data: 80%

Testing Data: 20%

Time Step: 60 days

Model Evaluation

Two evaluation metrics are used.

RMSE

Root Mean Squared Error (RMSE) measures the difference between actual and predicted values.

A lower RMSE indicates better prediction performance.

MAE

Mean Absolute Error (MAE) calculates the average absolute difference between actual and predicted values.

A lower MAE indicates better prediction performance.

Visualizations

The project generates the following graphs:

Historical Stock Price – Shows the historical closing price of Apple stock.

Actual vs Predicted Price – Compares actual stock prices with prices predicted by the LSTM model.

Training and Validation Loss – Shows how training and validation loss changes during model training.

Next-Day Forecast

After training, the model takes the latest 60 days of closing prices and predicts the next day's closing price.

Example:

======================================
NEXT DAY STOCK PRICE FORECAST
======================================

Predicted next day's closing price: XX.XX

The exact prediction will vary depending on the downloaded dataset and model training.

Technologies Used

Python

NumPy

Pandas

Matplotlib

Scikit-learn

TensorFlow / Keras

yfinance

Jupyter Notebook / VS Code

Installation

Install the required Python libraries using:

pip install yfinance tensorflow scikit-learn pandas numpy matplotlib

How to Run

Step 1: Open the Project

Open the project in VS Code or Jupyter Notebook.

Step 2: Install Dependencies

Run:

pip install yfinance tensorflow scikit-learn pandas numpy matplotlib

Step 3: Run the Notebook

Run the LSTM code from beginning to end.

The program will automatically download AAPL historical stock-price data using yfinance.

Step 4: View the Results

The program will display:

Dataset information

Missing-value information

Historical stock-price graph

LSTM model summary

Training progress

RMSE

MAE

Actual vs predicted graph

Training vs validation loss graph

Next-day predicted closing price

Project Structure

LSTM-Stock-Forecasting/
│
├── LSTM_Stock_Forecasting.ipynb
│
└── README.md

No CSV file is required because the stock data is downloaded automatically using yfinance.