#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 21:15:59 2024

@author: xiangsihan

Aim: Use different machine learning methods to forecast the daily return of S&P500, 
so that determine the trading signals.

Pay attention: Ignore Feature Engineering, which means the result may be bad
"""
############################### Preparation #################################

#Import relative packages
from pandas_datareader import data as pdr
import yfinance as yf
(
    yf
    .pdr_override()
)

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

#Import Packages of Machine Learning
# Loading Algorithm
from sklearn.linear_model import LinearRegression
# Regularization
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
# Decision Tree
from sklearn.tree import DecisionTreeRegressor

# ENSEMBLE
## Bagging (Bootstrapped Aggregation)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
## Boosting
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Support Vector Machine
from sklearn.svm import SVR
# K-Nearest Neighbor
from sklearn.neighbors import KNeighborsRegressor
# Multi-layer Perceptron (Neural Networks)
from sklearn.neural_network import MLPRegressor

#backtesting method
from sklearn.metrics import mean_squared_error


start_date = dt.datetime(2003, 1, 1)
end_date = dt.datetime(2023, 12, 31)

#Feature select
stock_ticker = ['AAPL', "AMZN", "MSFT", 'F']
currency_ticker = ["DEXJPUS", "DEXUSUK"]
index_ticker = ["^DJI", "^VIX", "^TNX"]

stock_data = pdr.get_data_yahoo(stock_ticker, start = start_date, end = end_date)
currency_data = pdr.get_data_fred(currency_ticker, start = start_date, end = end_date)
index_data = yf.download(index_ticker, start = start_date, end = end_date)

#Independent variable: S&P500
y_ticker = "^SPX"
y_data = yf.download(y_ticker, start = start_date, end = end_date)


############################### Data Wrangling #################################
y_data["log return"] = np.log(y_data["Adj Close"]).diff()

#Y
Y = y_data["log return"]

#Autocorrelated Variables (lag = 1-5)
X0 = pd.DataFrame()
for lag in range(1,6):
    col = "lag"+str(lag)
    X0[col] = y_data["log return"].shift(lag)

#Use daily return of other stocks price
daily_price = stock_data["Adj Close"]
X1 = daily_price.apply(lambda x: np.log(x).diff()).shift(1)

#Use FX rates
currency_data = currency_data.dropna()
X2 = currency_data.apply(lambda x: np.log(x).diff()).shift(1)

#Use other indices
index_data_1 = index_data["Adj Close"]
X3 = index_data_1.apply(lambda x: np.log(x).diff()).shift(1)

#Use technical indicators of S&P500
#Moving average (days = 21; 63; 252)
X4 = pd.concat([(y_data.loc[ : , ("Adj Close")]).rolling(window = i).mean() for i in [21, 63, 252]],axis = 1).shift(1)
X4.columns = ["SMA_21", "SMA_63", "SMA_252"]
#Exponential Moving average (days = 10; 30; 200)
X5 = pd.concat([(y_data.loc[ : , ("Adj Close")]).ewm(span = i, adjust = False).mean() for i in [10, 30, 200]],axis = 1).shift(1)
X5.columns = ["EWM_10", "EWM_30", "EWM_200"]
#RSI (days = 10; 30; 200)
#Function of calculating RSI
def calc_RSI(df,n):
    df["Return"] = df["Adj Close"].diff()
    df["up_sign"] = np.where(df["Return"]>0,df["Return"],0)
    df["down_sign"] = np.where(df["Return"]<0,abs(df["Return"]),0)
    df = df.dropna()
    df = df.reset_index(drop = False)
    df1 = df.iloc[0:n]
    average_loss = df1["down_sign"].mean()
    average_gain = df1["up_sign"].mean()
    RS = average_gain/average_loss
    RSI = 100-100/(1+RS)
    df.loc[n-1,"RSI"]=RSI
    for i in range(n,len(df)):
        average_loss = (average_loss*(n-1)+df.loc[i,"down_sign"])/n
        average_gain = (average_gain*(n-1)+df.loc[i,"up_sign"])/n
        RS = average_gain/average_loss
        RSI = 100-100/(1+RS)
        df.loc[i,"RSI"] = RSI
    return df
X6 = pd.DataFrame({"RSI10": calc_RSI(y_data,10)["RSI"], "RSI30": calc_RSI(y_data,30)["RSI"],"RSI200": calc_RSI(y_data,200)["RSI"]}).shift(1)
X6.index = calc_RSI(y_data,10)["Date"]

#combine all independent variables
X = pd.concat([X0,X1,X2,X3,X4,X5,X6],axis = 1)

#obtain the final dataframe
data = pd.concat([Y,X], axis = 1)
data = data.dropna()
############################### Performance Evaluation #################################
def strategy_performance(df):
    cum_ret = pd.DataFrame()
    sharpe_ratio = []
    sortino_ratio = []
    CAGR_ls = []
    MDD_ls = []
    for col_name in df.columns:
        strategy_cum_ret = df[col_name].dropna().cumsum().apply(np.exp)
        cum_ret = pd.concat([cum_ret, strategy_cum_ret], axis = 1)
        strategy_cum_change = cum_ret[col_name].pct_change().fillna(0)
        sharpe = np.sqrt(252) * (strategy_cum_change.mean() / strategy_cum_change.std())
        sortino_signal = np.where(strategy_cum_change < 0, strategy_cum_change, 0)
        sortino = np.sqrt(252) * strategy_cum_change.mean()/(((sortino_signal**2).sum()/len(sortino_signal))**0.5)
        days_CAGR = (strategy_cum_ret.index[-1] - strategy_cum_ret.index[0]).days
        CAGR = (strategy_cum_ret[-1]) / (strategy_cum_ret[0])**(365.0/days_CAGR) - 1
        sharpe_ratio.append(sharpe)
        sortino_ratio.append(sortino)
        CAGR_ls.append(CAGR)
        #Maximum Drawdown
        MAX_GROSS_PERFORMANCE = strategy_cum_ret.cummax()
        drawdown = strategy_cum_ret - MAX_GROSS_PERFORMANCE
        MDD = np.min(drawdown / MAX_GROSS_PERFORMANCE * 100)
        MDD_ls.append(MDD)
    final_result = pd.DataFrame({"Sharpe Ratio": sharpe_ratio,
                                     "CAGR": CAGR_ls,
                                     "Sortino Ratio": sortino_ratio,
                                     "Maximum Drawdown": MDD_ls})
    return final_result


############################### Model Building #################################



def model_test(models, X, Y, TimeLag):
#split train set and test set by using model data

    X_train, X_test = X[TimeLag[0]: TimeLag[1]], X[TimeLag[-2]: TimeLag[-1]]
    Y_train, Y_test = Y[TimeLag[0]: TimeLag[1]], Y[TimeLag[-2]: TimeLag[-1]]
    
    train_results = []
    test_results = []
    names = []
    cols = []
    
    backtest_signal = pd.DataFrame(Y_test)
    
    for name, model in models:
        names.append(name)
        #Train the data
        res = model.fit(X_train, Y_train)
        train_result = mean_squared_error(res.predict(X_train), Y_train)
        train_results.append(train_result)
        #Test the result
        test_result = mean_squared_error(res.predict(X_test), Y_test)
        test_results.append(test_result)
        #Trading Strategy
        predicted = res.predict(X_test)
        predicted = pd.DataFrame(predicted, index=Y_test.index)
        backtest_signal[name] = np.sign(predicted)
        
        cols.append(name)
    
    backtest_return = backtest_signal[['log return']]    
    for col in cols:
        backtest_return[f"return_{col}"] = backtest_return["log return"]*backtest_signal[col]
    backtest_return = backtest_return.rename(columns={'log return': 'Asset return'})
    
    for name in backtest_return.columns:
        plt.plot(np.exp(backtest_return[f"{name}"].cumsum()),
                 label = f"{name}")
    plt.legend(loc='upper left')
    #backtesting the portfolio performance
    
    fig = plt.figure(figsize = [8, 4])

    ind = np.arange(len(names))
    width = 0.30
    fig.suptitle\
        ("Comparing the Perfomance of Various Algorithms\
on the Training vs. Testing")

    ax = fig.add_subplot(111)

    (plt
     .bar(ind - width/2,
          train_results,
          width = width,
          label = "Errors in Training Set")
    )

    (plt
     .bar(ind + width/2,
          test_results,
          width = width,
          label = "Errors in Testing Set")
    )
    plt.legend()
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    plt.ylabel("Mean Squared Error (MSE)")
    plt.show() 
    
    return strategy_performance(backtest_return)
    

############################### Different Models #################################
Y_0 = data.iloc[:,0:1]
X_0 =data.iloc[:,1:]
date1 = dt.datetime(2003, 1, 1)
date2 = dt.datetime(2011, 12, 31)
date3 = dt.datetime(2013, 12, 31)
date4 = dt.datetime(2023, 12, 31)
time_lag1 = [date1,date2,date3]
time_lag2 = [date1, date3, date4]
#LASSO
models_Lasso = []
for i in range(10):
    models_Lasso.append((f"LASSO_{i}", Lasso(alpha = 0.01*(i+1))))

Lasso_test = model_test(models_Lasso, X_0, Y_0, time_lag1)

#Ridge
models_Ridge = []
for i in range(20):
    models_Ridge.append((f"RIDGE_{i}", Ridge(alpha = 0.01*(i+1))))
Ridge_test = model_test(models_Ridge, X_0, Y_0, time_lag1)

#SVM
models_SVR = []
svr1 = SVR(kernel = 'linear',epsilon=0.56,C=25) 
svr2 = SVR(kernel = 'poly',degree=4,epsilon=0.56,C=25)
svr3 = SVR(kernel = 'rbf',epsilon=0.56,C=25)
models_SVR.append(('SVR Linear', svr1))
models_SVR.append(('SVR Polynomial', svr2))
models_SVR.append(('SVR Sigmoid', svr3))
SVM_test = model_test(models_SVR, X_0, Y_0, time_lag1)
SVM_test[0]

#All models
models = []

# Regression and tree regression algorithms
models.append(("LR", LinearRegression()))
models.append(("RIDGE", Ridge(alpha = 0.01)))
models.append(("LASSO", Lasso(alpha = 0.01)))
models.append(("EN", ElasticNet()))
models.append(("CART", DecisionTreeRegressor()))
models.append(("KNN", KNeighborsRegressor()))
models.append(("SVR", SVR(kernel = 'linear',epsilon=0.56,C=25)))
# Ensemble models
# Bagging (Boostrap Aggregation)
models.append(("RFR", RandomForestRegressor()))
models.append(("ETR", ExtraTreesRegressor()))
# Boosting
models.append(("GBR", GradientBoostingRegressor(random_state = 2)))
models.append(("ABR", AdaBoostRegressor()))

model_test(models, X_0, Y_0, time_lag1)

############################### Out of Sample #################################
model_deploy = model_test(models, X_0, Y_0, time_lag2)
model_deploy
