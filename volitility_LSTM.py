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

file_name = 'kospi200f_805.csv'
item_name = 'kospi200f'
train_start = '2014-01-01'
train_end = '2018-11-30'
test_start = '2019-01-01'
test_end = '2019-11-19'

remove_columns = []
target_column = '종가(포인트)'
input_columns = []
target_type = 'diff'

model_name = 'transfer_LSTM'
channel = False

import gc
gc.collect()

target_alpha = 100
future_day = 20
n_timestep = 120
time_interval = 2
input_size = 805
n_unit = 800
batch_size = 32
learning_rate = 0.0005
n_iteration = 10000

if __name__ == "__main__":

    dataframe = util.read_datafile(file_name)
    df = dataframe.copy()
    df = prepro.normalization(df, 20, target_column)  # moving window normalization
    df = prepro.target_conversion(df, target_column, future_day, type=target_type)

    #calc current train, test date index
    current_train_start = df.loc[prepro.date_to_index(df, train_start), 'date']
    current_train_end = df.loc[prepro.date_to_index(df, train_end), 'date']
    current_test_start = df.loc[prepro.date_to_index(df, test_start), 'date']
    current_test_end = df.loc[prepro.date_to_index(df, test_start) + future_day - 1, 'date']

    #  각 transfer 구간의 예측값들을 합치기 위하여
    test_prediction = []

    while True:
        #early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)
        early_stopping = learn.EarlyStopping(patience=2, verbose=1)

        train_data, test_data = prepro.get_train_test_data(df, target_column, current_train_start, current_train_end,
                                                               current_test_start, current_test_end,
                                                               future_day, n_timestep, time_interval)
        # input_size, columns reset
        input_size = len(df.columns) - 2
        input_columns = df.columns.copy()

        train_x, train_y = prepro.get_LSTM_dataset(train_data, n_timestep, time_interval, input_size, future_day)
        test_x, test_y = prepro.get_LSTM_dataset(test_data, n_timestep, time_interval, input_size, future_day)

        model = models.LSTM(n_timestep,input_size,n_unit,regularizers_alpha=0.01,drop_rate=0.5)
        #keras.utils.plot_model(model, model_name+'_model_with_shape_info.png', show_shapes=True)

        #global_step = tf.train.get_or_create_global_step()
        global_step = tf.Variable(0, trainable=False)
        #lr_decay = tf.train.exponential_decay(learning_rate, global_step,
        #                                      train_input.shape[0]/batch_size*5, 0.5, staircase=True)
        lr_decay = tf.compat.v1.train.exponential_decay(learning_rate,global_step, 100000, 0.96, staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        for iteration in range(300):
            batch_input, batch_output = learn.next_random_batch(train_x, train_y, batch_size)
            batch_test_input, batch_test_output = learn.next_random_batch(test_x, test_y, len(test_x))

            #noise = 2*np.random.randn(batch_size,n_timestep,1)
            #batch_output = batch_output+noise
            #batch_input = encoder(train_input[idx])
            gradients = models.gradient(model, 'mean_square',  None, batch_input, batch_output)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if iteration %10 == 0:
                loss = models.loss_mean_square(model, batch_input, batch_output)
                train_MSE = models.evaluate(model, batch_input, batch_output)
                #test_MSE =  models.evaluate(model, test_input[:100], test_output[:100])
                test_MSE =  models.evaluate(model, batch_test_input, batch_test_output)

                print('iteration :', iteration, ' loss =', loss.numpy(), ' train MSE =', train_MSE.numpy(),' test MSE =', test_MSE.numpy())

            if iteration>100 and early_stopping.validate(test_MSE)==True:
                break

        p = learn.predict_batch_test(model, test_x, len(test_x))
        test_prediction.append(p)

        # 변동성 학습
        current_prediction = p[:, -1, -1].reshape((-1))
        current_output = tf.cast(test_y[:, -1, -1].reshape((-1)), tf.float32)
        v_model = models.LSTMno_seq(n_timestep, input_size, 100, regularizers_alpha=0.01)
        ones = tf.ones([20], tf.float32)
        for i in range(10):
            with tf.GradientTape() as tape:
                v_output = v_model(test_x, training=True)
                std = tf.reshape(v_output, [-1])
                prob = (1 / tf.sqrt(np.pi * std * std)) * tf.exp(
                    -tf.square(current_output - current_prediction) / tf.square(std))
                v_loss = tf.reduce_mean(-tf.math.log(prob))
                #v_loss = tf.reduce_mean(tf.square(1 - prob))
            v_gradients = tape.gradient(v_loss, v_model.trainable_variables)
            optimizer.apply_gradients(zip(v_gradients, v_model.trainable_variables))

        #v_output = v_model(test_x, training=False)
        cnt = 0
        for i in range(len(test_y)):
            if current_output[i] < current_prediction[i] + 1.96 * std[i] and current_output[i] > current_prediction[i] -1.96 * std[i]:
               cnt += 1
        print("real이 95% 신뢰구간안에 :", cnt / len(test_x))



        # escape from while
        if current_test_end == test_end:
            break

        #train, start dates shift
        current_train_start = df.loc[prepro.date_to_index(df, current_train_start) + future_day, 'date']
        current_train_end = df.loc[prepro.date_to_index(df, current_train_end) + future_day, 'date']
        current_test_start = df.loc[prepro.date_to_index(df, current_test_start) + future_day, 'date']
        if prepro.date_to_index(df, test_end) - prepro.date_to_index(df, current_test_start) < future_day:
            current_test_end = test_end
        else:
            current_test_end = df.loc[prepro.date_to_index(df, current_test_end) + future_day, 'date']

    test_prediction = np.concatenate(test_prediction, axis=0)
    train_prediction = learn.predict_batch_test(model, train_x,100)

    test_dates, test_base_prices, train_dates, train_base_prices = prepro.get_test_dates_prices(dataframe, test_start, test_end,
                                                          train_start, train_end, n_timestep, time_interval, future_day, target_column)

    # 전체 test_oouput 생성
    test_data = prepro.get_test_dataset(df, test_start, test_end,
                                            future_day, n_timestep, time_interval)
    _, test_y = prepro.get_LSTM_dataset(test_data, n_timestep, time_interval, input_size, future_day)

    result = GenerateResult(train_prediction,train_y,test_prediction,test_y, n_timestep, future_day)
    result.extract_last_output()
    result.convert_price(train_base_prices,test_base_prices,conversion_type=util.conversion)
    result.evaluation()
    result.table()
    result.save_result(model_name,item_name,n_unit,target_type,n_timestep,time_interval)
    result.save_visualization()
    result.save_model(model)