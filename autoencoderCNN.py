# Copyright 2018 Bimghi Choi. All Rights Reserved.
# autoencoderLSTM.py
# LSTM autoencoder --> ResLSTM 모델로 학습하고 test하여 결과 저장

# coding: utf-8

import models
import learn
from learn import GenerateResult
import math


from tensorflow import keras

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

import util
import preprocess as prepro


config = tf.cpmpat.v1.ConfigProto()
config.gpu_options.allow_growth = True

#tf.enable_eager_execution(config = config)
tf.compat.v1.enable_eager_execution(config=config)

import gc
gc.collect()

item_names = ['kospi200f']
train_start = util.train_start
train_end = util.train_end
test_start = util.test_start
test_end = util.test_end

remove_columns = util.remove_columns
target_column = util.target_column
input_columns = []
target_type = util.conversion

model_name = 'autoCNN_LSTM'
channel = False

target_alpha = 100
future_days = [20]
n_timesteps = [24]
time_intervals = [5]
input_size = util.config.input_size
n_unit = 50
batch_size = 32
learning_rate = 0.0005
n_iteration = 10000

if __name__ == "__main__":

    for item_name in item_names:
        dataframe = util.read_datafile(util.file_name)
        df = prepro.normalization(dataframe, 20, target_column)  # moving window normalization
        for future_day in future_days:
            df = prepro.target_conversion(df, target_column, future_day, type=target_type)
            for n_timestep in n_timesteps:
                for time_interval in time_intervals:
                    #early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)
                    early_stopping = learn.EarlyStopping()

                    train_data, test_data = prepro.get_train_test_data(df, train_start, train_end, test_start, test_end,
                                                                       future_day, n_timestep, time_interval)
                    # input_size, columns reset
                    input_size = len(df.columns) - 2
                    input_columns = df.columns.copy()

                    train_x, train_y = prepro.get_CLNN_dataset(train_data, n_timestep, time_interval, input_size, future_day)
                    test_x, test_y = prepro.get_CLNN_dataset(test_data, n_timestep, time_interval, input_size, future_day)

                    autoencoder = models.AutoEncoderCNN(n_timestep,input_size)
                    #keras.utils.plot_model(autoencoder, 'autoencoder_model_with_shape_info.png', show_shapes=True)
                    #train_input_encoder = np.r_[train_input,test_input]
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                    for iteration in range(1000):
                        batch_input, batch_output = learn.next_random_batch(train_x, train_x, batch_size)

                        gradients = models.gradient(autoencoder, 'mean_square', batch_input, batch_output)
                        optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

                        if iteration % 10 == 0:
                            loss = models.loss_mean_square(autoencoder, batch_input, batch_output)
                            train_MSE = models.evaluate(autoencoder, batch_input, batch_output)
                            test_MSE =  models.evaluate(autoencoder, test_x[:100], test_x[:100])

                            print('iteration :', iteration, ' loss =', loss.numpy(), ' train MSE =', train_MSE.numpy(),' test MSE =', test_MSE.numpy())


                    encoding = (keras.Model(inputs=autoencoder.input,outputs=autoencoder.get_layer('conv').output))
                    n_input = 38

                    model = models.ResLSTM2noise(n_timestep,n_input,n_unit,regularizers_alpha=0.01,drop_rate=0.3)
                    #keras.utils.plot_model(model, model_name+'_model_with_shape_info.png', show_shapes=True)

                    #global_step = tf.train.get_or_create_global_step()
                    global_step = tf.Variable(0, trainable=False)
                    #lr_decay = tf.train.exponential_decay(learning_rate, global_step,
                    #                                      train_input.shape[0]/batch_size*5, 0.5, staircase=True)
                    lr_decay = tf.compat.v1.train.exponential_decay(learning_rate,global_step, 100000, 0.96, staircase=True)
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)
                    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                    for iteration in range(500):
                        batch_input, batch_output = learn.next_random_batch(train_x, train_y, batch_size)
                        batch_test_input, batch_test_output = learn.next_random_batch(test_x, test_y, batch_size)
                        batch_input = tf.reshape(encoding(batch_input, training = False), [-1, n_timestep, n_input])
                        batch_test_input = tf.reshape(encoding(batch_test_input, training=False), [-1, n_timestep, n_input])
                        #noise = 2*np.random.randn(batch_size,n_timestep,1)
                        #batch_output = batch_output+noise
                        #batch_input = encoder(train_input[idx])
                        gradients = models.gradient(model, 'mean_square', batch_input, batch_output)
                        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                        if iteration %10 == 0:
                            loss = models.loss_mean_square(model, batch_input, batch_output)
                            train_MSE = models.evaluate(model, batch_input, batch_output)
                            #test_MSE =  models.evaluate(model, test_input[:100], test_output[:100])
                            test_MSE =  models.evaluate(model, batch_test_input, batch_test_output)

                            print('iteration :', iteration, ' loss =', loss.numpy(), ' train MSE =', train_MSE.numpy(),' test MSE =', test_MSE.numpy())

                        if iteration>500 and early_stopping.validate(test_MSE)==True:
                            break

                    train_encoder = tf.reshape(learn.predict_CNN_batch_test(encoding, train_x,100), [-1, n_timestep, n_input])
                    test_encoder = tf.reshape(learn.predict_CNN_batch_test(encoding, test_x,100), [-1, n_timestep, n_input])

                    test_prediction = learn.predict_batch_test(model, test_encoder,100)
                    train_prediction = learn.predict_batch_test(model, train_encoder,100)

                    test_dates, test_base_prices, train_dates, train_base_prices = prepro.get_test_dates_prices(dataframe, test_start, test_end,
                                                                          train_start, train_end, n_timestep, time_interval, future_day, target_column)

                    result = GenerateResult(train_prediction,train_y,test_prediction,test_y, n_timestep, future_day)
                    result.extract_last_output()
                    result.convert_price(train_base_prices,test_base_prices,conversion_type=util.conversion)
                    result.evaluation()
                    result.table()
                    result.save_result(model_name,item_name,n_unit,target_type,n_timestep,time_interval)
                    result.save_visualization()
                    result.save_model(model)