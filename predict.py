# Copyright 2018 Bimghi Choi. All Rights Reserved.

# _*_ coding: utf-8 _*_

import models
import learn
from learn import GenerateResult
import math


from tensorflow import keras

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0, 1'

import util
import preprocess as prepro

file_name = 'C:/Users/user/Desktop/DeepMoney_Category/DeepMoneyData/new/kospi200f_norm_200305.csv'
item_name = 'kospi200f'
test_start = '2019-03-01'
test_end = '2019-06-30'

remove_columns = []
target_column = '종가'
input_columns = []
target_type = 'diff'

model_name = 'transfer_LSTM'
channel = False

import gc
gc.collect()

target_alpha = 100
future_day = 2
n_timestep = 60
time_interval = 1
input_size = 2420
n_unit = 800
batch_size = 32
learning_rate = 0.0005
n_iteration = 10000

checkpoint_path = "model_2일후예측/pred2_trans20.ckpt"

if __name__ == "__main__":

    dataframe = util.read_datafile(file_name)
    df = dataframe.copy()
    #df = prepro.normalization(df, 20, target_column)  # moving window normalization
    df = prepro.target_conversion(df, target_column, future_day, type=target_type)

    # together with target column and without date column
    test_data = prepro.get_test_dataset(df, target_column, test_start, test_end,
                                                       future_day, n_timestep, time_interval)
    # input_size, columns reset
    input_size = len(df.columns) - 2

    test_x, test_y = prepro.get_LSTM_dataset(test_data, n_timestep, time_interval, input_size, future_day)
    train_x, train_y = test_x, test_y

    model = models.LSTM(n_timestep, input_size, n_unit, regularizers_alpha=0.01, drop_rate=0.5)
    model.compile(optimizer='adam',
                  loss='mse')
                  #callbacks=[cp_callback])
                  # metrics=['accuracy'])

    model.load_weights(checkpoint_path)
    test_prediction = model.predict(test_x)
    train_prediction = test_prediction

    test_dates, test_base_prices = prepro.get_test_only_dates_prices(
                                                                   dataframe, test_start, test_end,
                                                                   n_timestep, time_interval, future_day, target_column)
    train_dates, train_base_prices = test_dates, test_base_prices

    result = GenerateResult(train_prediction,train_y,test_prediction,test_y, n_timestep, future_day, test_dates)
    result.extract_last_output()
    result.convert_price(train_base_prices,test_base_prices,conversion_type=util.conversion)
    result.evaluation()
    result.table()
    result.save_result(model_name,item_name,n_unit,target_type,n_timestep,time_interval)
    result.save_visualization()
