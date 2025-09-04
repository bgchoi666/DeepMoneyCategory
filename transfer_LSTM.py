# Copyright 2018 Bimghi Choi. All Rights Reserved.
# ResLSTM2noise.py

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
item_name = 'kospi200f'
train_start = '2010-01-01'
train_end = '2019-01-05'
test_start = '2019-03-06'
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
future_day = 22
transfer_day = 22
n_timestep = 120
time_interval = 2
input_size = 1050
n_unit = 800
batch_size = 32
learning_rate = 0.0005
n_iteration = 10000

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

    #  각 transfer 구간의 예측값들을 합치기 위하여
    test_prediction = []
    model = models.LSTM(n_timestep, input_size, n_unit, regularizers_alpha=0.01, drop_rate=0.5)

    epochs = 300
    while True:
        # early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)
        early_stopping = learn.EarlyStopping(patience=2, verbose=1)

        train_data, test_data = prepro.get_train_test_data(df, target_column, remove_columns,
                                                               current_train_start, current_train_end,
                                                               current_test_start, current_test_end,
                                                               future_day, n_timestep, time_interval)
        # input_size, columns reset
        input_size = len(df.columns) - len(remove_columns)
        input_columns = df.columns.copy()

        train_x, train_y = prepro.get_LSTM_dataset(train_data, n_timestep, time_interval, input_size, future_day)
        test_x, test_y = prepro.get_LSTM_dataset(test_data, n_timestep, time_interval, input_size, future_day)

        #keras.utils.plot_model(model, model_name+'_model_with_shape_info.png', show_shapes=True)

        #global_step = tf.train.get_or_create_global_step()
        global_step = tf.Variable(0, trainable=False)
        #lr_decay = tf.train.exponential_decay(learning_rate, global_step,
        #                                      train_input.shape[0]/batch_size*5, 0.5, staircase=True)
        lr_decay = tf.compat.v1.train.exponential_decay(learning_rate,global_step, 100000, 0.96, staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        for iteration in range(epochs):
            batch_input, batch_output = learn.next_random_batch(train_x, train_y, batch_size)
            batch_test_input, batch_test_output = learn.next_random_batch(test_x, test_y, len(test_x))

            #noise = 2*np.random.randn(batch_size,n_timestep,1)
            #batch_output = batch_output+noise
            #batch_input = encoder(train_input[idx])
            gradients = models.gradient(model, 'mean_square', None, batch_input, batch_output)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if iteration % 10 == 0:
                train_MSE, _ = models.evaluate(model, batch_input, batch_output)
                #test_MSE =  models.evaluate(model, test_input[:100], test_output[:100])
                test_MSE, _ = models.evaluate(model, batch_test_input, batch_test_output)

                print('iteration :', iteration, ' train MSE =', train_MSE.numpy(),' test MSE =', test_MSE.numpy())

            if iteration > epochs / 2 and early_stopping.validate(test_MSE)==True:
                break
        epochs -= 50
        if epochs <= 0: epochs = 100

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

    train_base_prices = train_base_prices[:batch_size]
    train_dates = train_dates[:batch_size]
    train_x = train_x[:batch_size]
    train_y = train_y[:batch_size]

    train_prediction = learn.predict_batch_test(model, train_x,len(train_x))
    # 전체 test_oouput 생성
    test_data = prepro.get_test_dataset(df, test_start, test_end,
                                            future_day, n_timestep, time_interval)
    _, test_y = prepro.get_LSTM_dataset(test_data, n_timestep, time_interval, input_size, future_day)

    result = GenerateResult(train_prediction,train_y,test_prediction,test_y, n_timestep, future_day, transfer_day)
    result.extract_last_output()
    result.convert_price(train_base_prices,test_base_prices,conversion_type=util.conversion)
    result.evaluation()
    result.table()
    result.save_result(model_name,item_name,n_unit,target_type,n_timestep,time_interval)
    result.save_visualization()
    result.save_model(model)