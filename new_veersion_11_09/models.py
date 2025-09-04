# _*_ coding: utf-8 _*_

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
import numpy as np
import statistics

import os

import gc
gc.collect()

# drop rate 0.5, elu dense layer
class DenseLayer(Model):
    def __init__(self, n_units, drop_rate=.5, activation=tf.nn.tanh):
        super(DenseLayer, self).__init__()
        self.dense = layers.Dense(n_units,
                                  activation=activation,
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=keras.regularizers.l2(0.01))

        self.batchnorm = layers.BatchNormalization()
        self.drop = layers.Dropout(rate=drop_rate)

    def call(self, inputs, training=False):
        layer = self.dense(inputs)
        layer = self.batchnorm(layer)
        layer = tf.nn.elu(layer)
        layer = self.drop(layer)

        return layer


class LSTMLayer(Model):
    def __init__(self, n_units, drop_rate=.3, return_sequences=True):
        super(LSTMLayer, self).__init__()
        self.LSTM = tf.compat.v1.keras.layers.CuDNNLSTM(n_units,
                                                        return_sequences=return_sequences,
                                                        kernel_initializer='he_normal',
                                                        kernel_regularizer=keras.regularizers.l2(0.01))
        self.batchnorm = layers.BatchNormalization()
        self.drop = layers.Dropout(rate=drop_rate)
        # self.drop = layers.TimeDistributed(keras.layers.Dropout(rate = drop_rate))

    def call(self, inputs, training=False):
        layer = self.LSTM(inputs)
        layer = self.batchnorm(layer)
        layer = self.drop(layer)

        return layer


# 3개의 LSTM layers로 구성된 model dropout 없음
def LSTM(n_timestep, n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5):

    inputs = keras.Input(shape=(n_timestep, n_inputs))
    # dropout = layers.Dropout(rate = drop_rate)(inputs)
    lstm1 = layers.LSTM(n_units,
                        return_sequences=True,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(inputs)
    #batchnorm1 = layers.BatchNormalization()(lstm1)

    lstm2 = layers.LSTM(n_units,
                        return_sequences=True,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(lstm1)
    #batchnorm2 = layers.BatchNormalization()(lstm2)

    lstm3 = layers.LSTM(n_units,
                        return_sequences=True,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(lstm2)
    #batchnorm3 = layers.BatchNormalization()(lstm3)

    stacked_rnn_outputs = tf.reshape(lstm3, [-1, n_units])
    stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
    logits = tf.reshape(stacked_outputs, [-1, n_timestep, 1])

    return Model(inputs=inputs, outputs=logits)

# 3개의 LSTM layers로 구성된 model dropout 없음
def LSTM_sigm(n_timestep, n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5):

    inputs = keras.Input(shape=(n_timestep, n_inputs))
    # dropout = layers.Dropout(rate = drop_rate)(inputs)
    lstm1 = layers.LSTM(n_units,
                        return_sequences=False,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(inputs)
    #batchnorm1 = layers.BatchNormalization()(lstm1)

    #lstm2 = layers.LSTM(n_units,
    #                    return_sequences=True,
    #                    kernel_initializer='he_normal',
    #                    kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(lstm1)
    #batchnorm2 = layers.BatchNormalization()(lstm2)

    stacked_rnn_outputs = tf.reshape(lstm1, [-1, 10])
    stacked_outputs = keras.layers.Dense(units=1, activation='sigmoid')(stacked_rnn_outputs)
    logits = tf.reshape(stacked_outputs, [-1, n_timestep, 1])

    return Model(inputs=inputs, outputs=logits)

# 3개의 LSTM layers로 구성된 model dropout 없음
def LSTM_tanh(n_timestep, n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5):

    inputs = keras.Input(shape=(n_timestep, n_inputs))
    # dropout = layers.Dropout(rate = drop_rate)(inputs)
    lstm1 = layers.LSTM(n_units,
                        return_sequences=True,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(inputs)
    #batchnorm1 = layers.BatchNormalization()(lstm1)

    lstm2 = layers.LSTM(n_units,
                        return_sequences=True,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(lstm1)
    #batchnorm2 = layers.BatchNormalization()(lstm2)

    stacked_rnn_outputs = tf.reshape(lstm2, [-1, n_units])
    stacked_outputs = keras.layers.Dense(units=1, activation='tanh')(stacked_rnn_outputs)
    logits = tf.reshape(stacked_outputs, [-1, n_timestep, 1])

    return Model(inputs=inputs, outputs=logits)


# 최종 output -1 ~ 1
def DenseLayer(n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5):

    inputs = keras.Input(shape=(n_inputs))
    # dropout = layers.Dropout(rate = drop_rate)(inputs)
    Dense = layers.Dense(n_units,
                        activation='relu',
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(inputs)
    batchnorm1 = layers.BatchNormalization()(Dense)

    logits = layers.Dense(1,
                        activation='tanh',
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm1)
    #batchnorm2 = layers.BatchNormalization()(lstm2)
    #logits = tf.reshape(logits, [-1])
    
    return Model(inputs=inputs, outputs=logits)

# 최종 output linear
def DenseLayer_linear(n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5):

    inputs = keras.Input(shape=(n_inputs))
    # dropout = layers.Dropout(rate = drop_rate)(inputs)
    Dense = layers.Dense(n_units,
                        activation='relu',
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(inputs)
    batchnorm1 = layers.BatchNormalization()(Dense)

    logits = layers.Dense(1,
                        activation='relu', 
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm1)
    #batchnorm2 = layers.BatchNormalization()(lstm2)
    #logits = tf.reshape(logits, [-1])
    
    return Model(inputs=inputs, outputs=logits)

# 최종 output 0 ~ 1
def DenseLayer_sigmoid(n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5):

    inputs = keras.Input(shape=(n_inputs))
    # dropout = layers.Dropout(rate = drop_rate)(inputs)
    Dense = layers.Dense(n_units,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(inputs)
    batchnorm1 = layers.BatchNormalization()(Dense)

    logits = layers.Dense(1,
                        activation='sigmoid',
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm1)
    #batchnorm2 = layers.BatchNormalization()(lstm2)
    #logits = tf.reshape(logits, [-1])
    
    return Model(inputs=inputs, outputs=logits)

# 2개의 LSTM layer로 256feature encoding, 2개의 LSTM layer로 decoding
def AutoEncoderLSTM(n_timestep, n_inputs, regularizers_alpha=0.01, drop_rate=0.5, gpu=0):
    with tf.device('/cpu:0'):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        dropout = layers.Dropout(rate=drop_rate)(inputs)
        lstm1 = layers.LSTM(1024,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha), name='HiddenLayer1')(dropout)
        # batchnorm1 = layers.BatchNormalization()(lstm1)


        lstm2 = layers.LSTM(256,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha), name='HiddenLayer2')(lstm1)
        # batchnorm2 = layers.BatchNormalization(name = 'encoding')(lstm2)



        lstm3 = layers.LSTM(1024,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha), name='HiddenLayer3')(lstm2)
        # batchnorm3 = layers.BatchNormalization()(lstm3)

        lstm4 = layers.LSTM(n_inputs,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(lstm3)
        # batchnorm4 = layers.BatchNormalization()(lstm4)


        return Model(inputs=inputs, outputs=lstm4)

def AutoEncoderCNN(n_steps, n_inputs):

      inputs = keras.Input(shape=(n_steps, n_inputs, 1))
      conv = tf.keras.layers.Conv2D(
          filters=1, kernel_size=(1, 100), strides=(1, 10), activation='relu', padding='same', name='conv')(inputs)

      convT = tf.keras.layers.Conv2DTranspose(
          filters=1,
          kernel_size=(1, 100),
          strides=(1, 9),
          padding="SAME",
          activation='relu')(conv)
      # No activation
      outputs = tf.keras.layers.Conv2DTranspose(
          filters=1, kernel_size=(1,32), strides=(1, 1), padding="VALID")(convT)

      return Model(inputs=inputs, outputs=outputs)


def loss_mean_square(model, input_data, output_data):
    logits = model(input_data, training=True)
    loss = tf.reduce_mean(tf.square(logits - output_data))
    return loss


def loss_cross_entropy(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels))
    return loss


def loss_add_sharp_ratio(model, base_data, input_data, output_data):
    prediction_data = model(input_data, training=True)
    signs = (base_data - prediction_data) * (base_data - output_data)
    profit_rates = signs * abs((output_data - base_data)) / base_data # 수익률 list
    profit_rate_mean = tf.math.reduce_mean(profit_rates) # 수익률 평균
    std = tf.math.reduce_std(profit_rates) # 수익률의 표준 편차
    sharp_ratio = profit_rate_mean / std
    loss = tf.reduce_mean(tf.square(prediction_data - output_data)) + sharp_ratio
    return loss


def loss_add_variance(model, base_data, input_data, output_data):
    prediction_data = model(input_data, training=True)
    signs = (base_data - prediction_data) * (base_data - output_data)
    profit_rates = signs * abs((output_data - base_data)) / base_data # 수익률 list
    var = statistics.variance(profit_rates) # 수익률의 분산
    loss = tf.reduce_mean(tf.square(prediction_data - output_data)) + var
    return loss

def loss_prediction_interval(model, input_data, prediction, real):
    std = tf.reshape(model(input_data, training=False)[:, -1, -1], [-1])
    prob = (1 / tf.sqrt(2 * np.pi * std * std)) * tf.exp(-tf.square(prediction - real) / tf.square(std))
    loss = -tf.math.log(prob)
    return loss

def gradient(model, type, input_data, output_data):
    with tf.GradientTape() as tape:
        if type == "cross_entropy" : loss = loss_mean_square(model, input_data, output_data)
        elif type == "add_sharp_ratio" : loss = loss_mean_square(model, input_data, output_data)
        elif type == "add_variance" : loss = loss_mean_square(model, input_data, output_data)
        else : loss = loss_mean_square(model, input_data, output_data)
    return tape.gradient(loss, model.trainable_variables)

def compute_apply_gradients(model, loss, optimizer):
    gradients = gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def evaluate(model, input_data, output_data, dropout=False):
    logits = model(input_data, training=dropout)
    MSE = tf.reduce_mean(tf.square(logits - output_data))
    return MSE

