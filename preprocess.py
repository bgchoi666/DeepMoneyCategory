# Copyright 2018 Bimghi Choi. All Rights Reserved.
# preprocess.py
# data frame으로부터 이상치, 결측치 처리, normalization, 불필요한 column제거 등 작업 수행한 후 train, test dataset로 분리한여 반환

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import math
import util
import datetime

# training test dates are converted to the indices
def date_to_index(df,date):

  # start : the predicted date,  start_index : the date to try to predict
  index = df[df['date'] <= date].index.max()

  return index

# dataframe으로부터 LSTM의 input으로 들어가는 dataset을 만들어 반환
def get_train_test_data(raw_df, target_column,remove_columns, train_start, train_end, test_start, test_end, predict_term, num_steps, step_interval):

      train_start_index = date_to_index(raw_df, train_start)
      train_end_index = date_to_index(raw_df, train_end)

      test_start_index = date_to_index(raw_df, test_start) # the starting date of predicted dates (y values)
      test_end_index = date_to_index(raw_df, test_end)

      # check date error
      if math.isnan(train_start_index):
        print('============== illegal train start date ==============')
        exit(1)
      if math.isnan(train_end_index):
        print('============== illegal train end date ==============')
        exit(1)
      if math.isnan(test_start_index):
        print('============== illegal test start date ==============')
        exit(1)
      if math.isnan(test_end_index):
        print('============== illegal test end date ==============')
        exit(1)
      #if test_start_index - predict_term <= train_end_index:
      #  print('============== train end, test start inconsistency error ==============')
      #  exit(1)

      # 날짜 column 등 제거
      df = raw_df.drop(remove_columns, axis=1, inplace=False)
      df = pd.concat([df, raw_df[target_column]], axis=1)

      # create train data
      train_data = df.values[train_start_index : train_end_index  + 1, :]

      # create test data
      test_data = df.values[test_start_index - predict_term - step_interval * (num_steps-1): test_end_index + 1, :]

      return train_data, test_data

def get_train_dataset(raw_df, target_column, train_start, train_end, predict_term, num_steps, step_interval):

    train_start_index = date_to_index(raw_df, train_start)
    train_end_index = date_to_index(raw_df, train_end)

    # check date error
    if math.isnan(train_start_index):
        print('============== illegal train start date ==============')
        exit(1)
    if math.isnan(train_end_index):
        print('============== illegal train end date ==============')
        exit(1)

    # 날짜 컬럼 제거, target column 맨 뒤로
    df = raw_df.drop(['date', target_column], axis=1, inplace=False)
    df = pd.concat([df, raw_df[target_column]], axis=1)

    # create train data
    train_data = df.values[train_start_index: train_end_index + 1, :]

    return train_data

def get_test_dataset(raw_df, target_column, test_start, test_end, predict_term, num_steps, step_interval):

    test_start_index = date_to_index(raw_df, test_start)  # the starting date of predicted dates (y values)
    test_end_index = date_to_index(raw_df, test_end)

    if math.isnan(test_start_index):
        print('============== illegal test start date ==============')
        exit(1)
    if math.isnan(test_end_index):
        print('============== illegal test end date ==============')
        exit(1)

    # 날짜 column 제거
    df = raw_df.drop(['date', target_column], axis=1, inplace=False)
    df = pd.concat([df, raw_df[target_column]], axis=1)

    # create test data
    test_data = df.values[test_start_index - predict_term - step_interval * (num_steps - 1): test_end_index + 1, :]

    return test_data

# dataset를 step_interval, steps에 따라 재구성한 후 input, target으로 분리하여 반환
def get_LSTM_dataset(dataset, num_steps, step_interval, input_size, predict_term):
  # split input and target
  dataX, dataY = [], []

  # calc the number of time series data
  size = len(dataset) - (num_steps-1) * step_interval - predict_term

  for i in range(size):
    input_list = list(range(i, i + num_steps * step_interval, step_interval))
    a = dataset[input_list, :input_size].astype(float)
    dataX.append(a)
    b = dataset[input_list, input_size: ].astype(float)
    dataY.append(b)

  # --> 3-D array dimension
  x = np.reshape(dataX, (-1, num_steps, input_size))
  y = np.reshape(dataY, (-1, num_steps, dataset.shape[1] - input_size))

  return x, y

def get_CNN_dataset(dataset, num_steps, step_interval, input_size, predict_term):
    # split input and target
    dataX, dataY = [], []

    # calc the number of time series data
    size = len(dataset) - (num_steps - 1) * step_interval - predict_term

    # label b 값은 마지막 step(최근 일자)의 target values
    for i in range(size):
        input_list = list(range(i, i + num_steps * step_interval, step_interval))
        a = dataset[input_list, :input_size].astype(float)
        dataX.append(a)
        b = dataset[num_steps - 1, input_size].astype(float)
        dataY.append(b)

    # --> 4-D array dimension
    x = np.reshape(dataX, (-1, num_steps, input_size, 1))
    y = np.reshape(dataY, (-1))

    return x, y

# CNN으로 input을 받아 변수 압축 후 LSTM으로 넘김. 따라서 input data (x)는 CNN type, output은 LSTM type
def get_CLNN_dataset(dataset, num_steps, step_interval, input_size, predict_term):
  # split input and target
  dataX, dataY = [], []

  # calc the number of time series data
  size = len(dataset) - (num_steps-1) * step_interval - predict_term

  for i in range(size):
    input_list = list(range(i, i + num_steps * step_interval, step_interval))
    a = dataset[input_list, :input_size].astype(float)
    dataX.append(a)
    b = dataset[input_list, input_size: ].astype(float)
    dataY.append(b)

  # --> 3-D array dimension
  x = np.reshape(dataX, (-1, num_steps, input_size, 1))
  y = np.reshape(dataY, (-1, num_steps, dataset.shape[1] - input_size))

  return x, y

# 예측을 시도하는 날짜, 종가, 예측의 target이 되는 날짜, 종가 list 반환
def get_test_dates_prices(df, test_start, test_end, train_start, train_end, n_steps, time_interval, predict_term, target_column):
  test_start_index = date_to_index(df, test_start)
  test_end_index = date_to_index(df, test_end)
  train_start_index = date_to_index(df, train_start) + (n_steps - 1) * time_interval + predict_term
  train_end_index = date_to_index(df, train_end)

  #days = int(predict_term / 5) * 7 + predict_term % 5
  #test_future_dates = []

  test_dates = list(df.loc[test_start_index : test_end_index, 'date'])
  #for i in range(len(test_dates)):
  #  test_future_dates.append((datetime.datetime.strptime(test_dates[i], "%Y-%m-%d") + datetime.timedelta(days=days)).strptime("%Y-%m-%d"))
  train_dates = df.loc[train_start_index: train_end_index, 'date']
  test_base_prices = list(map(float, df.loc[test_start_index - predict_term: test_end_index - predict_term, target_column]))
  train_base_prices = list(map(float, df.loc[train_start_index - predict_term: train_end_index - predict_term, target_column]))

  return test_dates, test_base_prices, train_dates, train_base_prices

# 예측을 시도하는 날짜, 종가, 예측의 target이 되는 날짜, 종가 list 반환
def get_test_only_dates_prices(df, test_start, test_end, n_steps, time_interval, predict_term, target_column):
  test_start_index = date_to_index(df, test_start)
  test_end_index = date_to_index(df, test_end)

  test_dates = df.loc[test_start_index : test_end_index, 'date']
  test_base_prices = list(map(float, df.loc[test_start_index - predict_term: test_end_index - predict_term, target_column]))

  return test_dates, test_base_prices

# 주어진 컬럼 제거 + 절반 이상 nan인 컬럼 제거, 나머지 nan 0으로 대체
def remove_columns(df, columns, thresh=50):
    df = df.drop(columns, axis=1)
    df = df.dropna(axis=1, how='any', thresh=thresh)
    return df

# window size로 standard normalization x'(t) = (x(t) - rolling(window_size).mean) / rolling(window_size).std
def normalization(df, window_size, target_column):
  df = remove_columns(df, util.remove_columns)
  #기호 제거
  df.replace(',', '', inplace=True)  # 숫자 사이에 쉼표 제거
  df.replace("!", "", inplace=True)
  df.replace("@", "", inplace=True)
  df.replace('$', '', inplace=True)  # 숫자 사이에 쉼표 제거
  df.replace("%", "", inplace=True)
  df.replace("^", "", inplace=True)
  df.replace("&", "", inplace=True)
  df.replace("*", "", inplace=True)
  df.replace('(', '', inplace=True)  # 숫자 사이에 쉼표 제거
  df.replace(")", "", inplace=True)
  df.replace("|", "", inplace=True)

  df = df.fillna(method='ffill') # 비어 있는 값 이전 값으로 채움
  not_input_columns = [target_column, 'date']
  input_df = df.drop(not_input_columns, axis=1) # target column 분리
  idf = (input_df - input_df.rolling(window_size).mean()) / input_df.rolling(window_size).std() #window moving normalize

  # nan, infitiy 0으로 대체
  idf.replace(np.NaN, 0, inplace=True)
  idf.replace(np.inf, 0, inplace=True)
  idf.replace(np.NINF, 0, inplace=True)

  # 70% 이상 0인 컬럼 제거
  for col in idf.columns:
      l_0 = len(idf.loc[idf[col] == 0, col])
      l = len(idf[col])
      if l_0 > l * 0.7:
          idf.drop(col, axis=1, inplace=True)

  norm_df = pd.concat([idf, df[not_input_columns]], axis=1).reset_index(drop=True)
  return norm_df

# target conversion - 'rate' : (y(t+predict_term) - y(t)) / y(t)
# target conversion - 'diff' : y(t+predict_term) - y(t)
#  target conversion - 'logdiff' : log(y(t+predict_term) - y(t))
def target_conversion(df, target_name, predict_term, type='diff'):
    if type == 'rate':
        for i in range(len(df[target_name]) - predict_term):
            df.loc[i, target_name] = (df.loc[i + predict_term, target_name] - df.loc[i, target_name])\
                                 / df.loc[i, target_name]*100
    elif type == 'diff':
        for i in range(len(df[target_name]) - predict_term):
            df.loc[i, target_name] = df.loc[i + predict_term, target_name] - df.loc[i, target_name]
    elif type == 'logdiff':
        for i in range(len(df[target_name]) - predict_term):
            df.loc[i, target_name] = np.log(df.loc[i + predict_term, target_name] - df.loc[i, target_name])
    else:
        print('target conversion error')
        exit(0)
    for i in range(len(df[target_name]) - predict_term, len(df[target_name])):
        df.loc[i, target_name] = 0.0
    return df

