# Copyright 2018 Bimghi Choi. All Rights Reserved.
# LSTM.py

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


import gc
gc.collect()

file_name = 'C:/Users/user/Desktop/DeepMoney_Category/DeepMoneyData/new/kospi200f_norm_200305.csv'
item_name = 'kospi200f_1050'
train_start = '2016-01-01'
train_end = '2018-09-30'
test_start = '2019-01-01'
test_end = '2020-03-05'

remove_columns = ['date', '종가', '시가']
target_column = '종가'
input_columns = []
target_type = 'diff'

model_name = 'transfer_LSTM'
channel = False

import gc
gc.collect()

target_alpha = 100
future_day = 60
transfer_day = 21
n_timestep = 120
time_interval = 2
input_size =1050
n_unit = 800
batch_size = 32
learning_rate = 0.0005
n_iteration = 10000

checkpoint_path = "model_60일후예측/pred60_trans21.ckpt"

if __name__ == "__main__":

    dataframe = util.read_datafile(file_name)
    df = dataframe.copy()
    #df = prepro.normalization(df, 20, target_column)  # moving window normalization
    df = prepro.target_conversion(df, target_column, future_day, type=target_type)

    #calc current train, test date index
    current_train_start = df.loc[prepro.date_to_index(df, train_start), 'date']
    current_train_end = df.loc[prepro.date_to_index(df, train_end), 'date']
    current_test_start = df.loc[prepro.date_to_index(df, test_start), 'date']
    current_test_end = df.loc[prepro.date_to_index(df, test_start) + transfer_day - 1, 'date']

    # 체크포인트 콜백 만들기
    #checkpoint_path = "models/transfer_STM2/{epoch:04d}.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
    #                                                 save_weights_only=True,
    #                                                 verbose=1)

    # 모델 생성성
    model = models.LSTM(n_timestep, input_size, n_unit, regularizers_alpha=0.01, drop_rate=0.5)
    model.compile(optimizer='adam',
                  loss='mse')
                  #callbacks=[cp_callback])
                  # metrics=['accuracy'])
    model.save_weights(checkpoint_path)

    # test_MSE가 더이상 줄어들지 않을 떄 조기 종료
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=2, verbose=1)

    #  각 transfer 구간의 예측값들을 합치기 위하여
    test_prediction = []

    while True:
        # early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)
        #early_stopping = learn.EarlyStopping(patience=2, verbose=1)

        train_data, test_data = prepro.get_train_test_data(df, target_column, remove_columns,
                                                               current_train_start, current_train_end,
                                                               current_test_start, current_test_end,
                                                               future_day, n_timestep, time_interval)
        # input_size, columns reset
        input_size = len(df.columns) - len(remove_columns)
        input_columns = df.columns.copy()

        train_x, train_y = prepro.get_LSTM_dataset(train_data, n_timestep, time_interval, input_size, future_day)
        test_x, test_y = prepro.get_LSTM_dataset(test_data, n_timestep, time_interval, input_size, future_day)

        model.load_weights(checkpoint_path)
        model.fit(train_x, train_y, batch_size=batch_size, epochs=12, callbacks=[early_stopping], validation_data=(test_x, test_y))
        model.save_weights(checkpoint_path)

        test_MSE, logits = models.evaluate(model, test_x, test_y)
        print('test_MSE = ', test_MSE)

        test_prediction.append(logits)

        print(current_test_start + " ~ " + current_test_end + " test finished")

        # escape from while
        if current_test_end == test_end:
            break

        #train, start dates shift
        current_train_start = df.loc[prepro.date_to_index(df, current_train_start) + transfer_day, 'date']
        current_train_end = df.loc[prepro.date_to_index(df, current_train_end) +  transfer_day, 'date']
        current_test_start = df.loc[prepro.date_to_index(df, current_test_start) +  transfer_day, 'date']
        if prepro.date_to_index(df, test_end) - prepro.date_to_index(df, current_test_start) < transfer_day:
            current_test_end = test_end
        else:
            current_test_end = df.loc[prepro.date_to_index(df, current_test_end) + transfer_day, 'date']

    test_prediction = np.concatenate(test_prediction, axis=0)


    test_dates, test_base_prices, train_dates, train_base_prices = prepro.get_test_dates_prices(dataframe, test_start, test_end,
                                                          train_start, train_end, n_timestep, time_interval, future_day, target_column)

    #train_base_prices = train_base_prices[250:350]
    #train_dates = train_dates[250:350]
    #train_x = train_x[250:350]
    #train_y = train_y[250:350]

    train_prediction = learn.predict_batch_test(model, train_x,len(train_x))
    # 전체 test_oouput 생성
    test_data = prepro.get_test_dataset(df, target_column, test_start, test_end,
                                            future_day, n_timestep, time_interval)
    _, test_y = prepro.get_LSTM_dataset(test_data, n_timestep, time_interval, input_size, future_day)

    result = GenerateResult(train_prediction,train_y,test_prediction,test_y, n_timestep, future_day, test_dates)
    result.extract_last_output()
    result.convert_price(train_base_prices,test_base_prices,conversion_type=util.conversion)
    result.evaluation()
    result.table()
    result.save_result(model_name,item_name,n_unit,target_type,n_timestep,time_interval)
    result.save_visualization()
    result.save_model(model)