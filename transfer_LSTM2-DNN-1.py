# Copyright 2018 Bimghi Choi. All Rights Reserved.
# ResLSTM2noise.py

# _*_ coding: utf-8 _*_

from tensorflow import keras
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='0, 1'

import util
import preprocess as prepro
import models
import learn
from learn import GenerateResult
import math
import datetime
import os
import random

import gc

file_name = 'Data/kospi200f_44_0129.csv'
item_name = 'kospi200f_44'
train_start = '2000-02-28'
train_end = '2017-12-20'
train_rate = 0.25
fix_valid = False
valid_start = '2017-01-04'
valid_end = '2018-09-30'
test_start = '2018-01-04'
test_end = '2020-12-31'

remove_columns = ['date', '시가', '고가', '저가']
target_column = '종가'
input_columns = []
target_type = 'rate'

model_name = 'transfer_LSTM2-DNN-1'
channel = False

target_alpha = 100
future_day = 5

input_size = 44
n_unit = 100
batch_size = 20
learning_rate = 0.0005

epochs=3

alpha = 0.05
beta = 0.01

max_repeat_cnt = 2500

if fix_valid:
    comment = "train: " + train_start + "~" + train_end + ", train data중 " + str(train_rate) + " 사용," +  "valid: "  + valid_start + "~" + valid_end + ", test: " + test_start + "~" + test_end
else:
    comment = "train: " + train_start + "~" + train_end + ", train data중 " + str(train_rate) + " 사용," +  "valid data는 train 중 20%, test: " + test_start + "~" + test_end

checkpoint_path = model_name + "/input44_batch20_unit100_DNN.ckpt"
checkpoint_path_best = model_name + "/input44_batch20_unit100_DNN_best.ckpt"

#model = models.LSTM(n_timestep,input_size,n_unit,regularizers_alpha=0.01,drop_rate=0.5)

dataframe = pd.read_csv(file_name, encoding='euc-kr')
raw_df = dataframe.copy()

def target_conversion(df, target_name, predict_term, type='diff'):
    if type == 'rate':
        for i in range(len(df[target_name]) - predict_term):
            df.loc[i, target_name] = (df.loc[i + predict_term, target_name] - df.loc[i, target_name]) \
                                     / df.loc[i, target_name] * target_alpha
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


df = target_conversion(raw_df, target_column, future_day, type=target_type)

train_start_index = max(df.loc[df['date'] <= train_start].index)
train_end_index = max(df.loc[df['date'] <= train_end].index)
base_prices = tf.reduce_mean(df.loc[train_start_index:train_end_index + 1, '종가'])



# @tf.function
def loss_add(targets, preds):
    preds = tf.reshape(preds, [-1])
    targets = tf.reshape(targets, [-1])

    if alpha != 0:
        # add RRL cost - maximize downside sharp ratio

        # 1 if (pred - base) * (target - base) > 0, -1 otherwise
        F = tf.math.sign(targets * preds)
        F = tf.reshape(F, [-1])

        # calc returns from each step in batches
        R = tf.math.divide(tf.math.multiply(tf.math.abs(targets), (F - 0.00003)), base_prices)
        R = tf.reshape(R, [-1])

        # calc downside sharp ratio

        # downside returns
        DR = tf.minimum(0.0, R)
        DR = tf.reshape(DR, [-1])

        # calc. downside sharp ratio
        # s = []
        # for i in range(batch_size):
        #   std =  tf.keras.backend.std(DR[i, :, 0])
        #   s.append(tf.reduce_mean(R[i, :, 0])/tf.maximum(0.01, std))

        # calc. downside sharp ratio
        loss1 = tf.reduce_mean(R) / (tf.keras.backend.std(DR) + 0.001)
    else:
        loss1 = 0

    """
    # average profits, loss
    avg_plusR = [0.0]
    avg_minusR = [0.0]

    global num_of_profits
    global num_of_losses

    num_of_profits = 0
    num_of_losses = 0

    for i in range(batch_size):
        res = tf.cond(R[i, num_steps - 1, 0] > 0, lambda: return_one(), lambda: return_zero())
        if res == 1:
            avg_plusR.append(R[i, num_steps - 1, 0])
        else:
            avg_minusR.append(R[i, num_steps - 1, 0])
    avg_profit = tf.reduce_mean(avg_plusR) 
    avg_loss = tf.reduce_mean(avg_minusR) 
    """

    if beta != 0:
        # compute maximum drawdown

        # accm_profit = [0.0]
        # for i in range(batch_size):
        #    for j in range(num_steps):
        #        r = tf.cond((predict_prices[i, num_steps-1, 0] - base_prices[i, num_steps-1, 0]) *
        #                   (target_prices[i, num_steps-1, 0] - base_prices[i, num_steps-1, 0]) > 0,
        #                   lambda: return_one(),
        #                   lambda: return_zero())
        #        if r == 1: accm_profit.append(accm_profit[i*num_steps + j] + tf.abs(target_prices[i, j, 0] - base_prices[i, j, 0]))
        #        else:      accm_profit.append(accm_profit[i*num_steps + j] - tf.abs(target_prices[i, j, 0] - base_prices[i, j, 0]))

        accm_profit = [0.0 for i in range(batch_size)]
        for i in range(batch_size):
            if i == 0:
                accm_profit[0] = tf.sign(preds * targets) * tf.math.abs(targets)
            else:
                accm_profit[i] = accm_profit[i - 1] + tf.sign(preds * targets) * tf.math.abs(targets)
        loss2 = (tf.reduce_max(accm_profit) - tf.reduce_min(accm_profit)) / batch_size
    else:
        loss2 = 0

    return beta * loss2 - alpha * loss1


@tf.function
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred)) + loss_add(y_true, y_pred)


model = tf.keras.Sequential([
    tf.keras.Input(shape=(input_size)),
    tf.keras.layers.Dense(n_unit, activation='relu'),
    tf.keras.layers.Dense(int(n_unit/2), activation='relu'),
    tf.keras.layers.Dense(1)
])

#cp_callback = tf.keras.callbacks.ModelCheckpoint(
#    checkpoint_path, verbose=1, save_weights_only=True,
    # 다섯 번째 에포크마다 가중치를 저장합니다
#    save_freq=5)

model.compile(optimizer='adam',
              loss='mse',
              #callbacks=[cp-callback]
              metrics=['accuracy'])
model.save_weights(checkpoint_path)

# 날짜 column 제거, target column 맨 뒤로
df = df.drop(remove_columns, axis=1, inplace=False)


def train():
    # train term index
    train_start_index = max(dataframe.loc[dataframe['date']<=train_start].index)
    train_end_index = max(dataframe.loc[dataframe['date']<=train_end].index)

    #ceate train data
    train_data = df.values[train_start_index : train_end_index + 1, :]

    # valid term index
    valid_start_index = max(dataframe.loc[dataframe['date']<=valid_start].index)
    valid_end_index = max(dataframe.loc[dataframe['date']<=valid_end].index)

    #ceate train data
    valid_data = df.values[valid_start_index : valid_end_index + 1, :]

    valid_x = valid_data[:, :44]
    valid_y = valid_data[:, 44]

    repeat_cnt = 0
    pre_accu = 0  # previous average accuracy

    while repeat_cnt < max_repeat_cnt:

        gc.collect()

        repeat_cnt += 1

        model.load_weights(checkpoint_path)

        train_x = train_data[:, :44]
        train_y = train_data[:, 44]

        # train data중 50%를 valid data로 사용 . . .
        train_x, _, train_y, _ = train_test_split(train_x, train_y, train_size=train_rate)

        # train data중 20%를 valid data로 사용 . . .
        if not fix_valid:
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)

        if repeat_cnt > 1:
            train_x, _, train_y, _ = train_test_split(np.concatenate([train_x, best_x]), np.concatenate([train_y, best_y]),
                                                      train_size=0.5)

        early_stopping = tf.keras.callbacks.EarlyStopping(patience=2, verbose=0)
        model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=epochs, callbacks=[early_stopping],
                  validation_data=(valid_x, valid_y))

        # test_MSE = model.evaluate(test_x, test_y)
        prediction = model.predict(valid_x)
        prediction_MSE = sum((valid_y - prediction.reshape((-1))) ** 2) / len(valid_y)
        # print('prediction_MSE =', prediction_MSE)

        # calculate accuracy
        temp = tf.math.multiply(np.sign(valid_y), prediction.reshape(-1))
        accu = tf.reduce_sum(list(map(lambda x: 1 if x > 0 else 0, temp))) / len(temp)
        if repeat_cnt % 100 == 0: print("반복 횟수 : " + str(repeat_cnt) + ' accuracy = ' + str(accu))

        if accu > pre_accu:
            #model.fit(valid_x, valid_y, batch_size=batch_size, epochs=epochs, verbose=0)
            model.save_weights(checkpoint_path_best)

            best_x = train_x
            best_y = train_y

            print("best accuracy " + str(accu))
            pre_accu = accu

    print("best accuracy " + str(pre_accu))

    return best_x, best_y

def test(best_x, best_y):
    #model.load_weights(checkpoint_path)
    #model.fit(best_x, best_y, batch_size=batch_size, epochs=3, )

    test_start_index = max(dataframe.loc[dataframe['date'] <= test_start].index)
    test_end_index = max(dataframe.loc[dataframe['date'] <= test_end].index)

    # create test data
    test_data = df.values[test_start_index: test_end_index + 1, :]

    test_x = test_data[:, :44]
    test_y = test_data[:, 44]

    test_dates = dataframe.loc[test_start_index + future_day: test_end_index + future_day, 'date']
    test_base_prices = list(
        map(float, dataframe.loc[test_start_index: test_end_index, target_column]))

    model.load_weights(checkpoint_path_best)
    test_prediction = model.predict(test_x).reshape(-1)

    # calculate accuracy
    temp = tf.math.multiply(np.sign(test_y), test_prediction.reshape(-1))
    accu = tf.reduce_sum(list(map(lambda x: 1 if x > 0 else 0, temp))) / len(temp)
    print('accuracy = ', accu)

    return test_prediction, test_y, test_dates, test_base_prices

# 변환된 target 값을 원 상태로 복원
def back_to_price(pred_values, base_prices, conversion_type='rate'):
    if conversion_type == 'rate':
        restored_price = (pred_values/target_alpha+1)*base_prices
    elif conversion_type == 'diff':
        restored_price = pred_values + base_prices
    elif conversion_type == 'logdiff':
        restored_price = np.exp(pred_values) + base_prices
    return list(restored_price)

def save(test_prediction, test_y, test_dates, test_base_prices):
    result = GenerateResult(test_prediction, test_y, test_prediction, test_y, 0, future_day, test_dates)
    #result.convert_price(test_base_prices, test_base_prices, conversion_type=target_type)

    result.train_predict_price = back_to_price(result.test_pred, test_base_prices, target_type)
    result.test_predict_price = back_to_price(result.test_pred, test_base_prices, target_type)
    # 실재값 복원
    result.train_output_price = back_to_price(result.test_output, test_base_prices, target_type)
    result.test_output_price = back_to_price(result.test_output, test_base_prices, target_type)

    result.evaluation()
    result.table()
    result.save_result(model_name, item_name, n_unit, target_type, 0, future_day, comment)
    result.save_visualization()
    # result.save_model(model, model_name)
    # model_path = 'models/'+model_name+'/'+result.info
    # if not os.path.isdir(model_path): os.mkdir(model_path)
    # keras.models.save_model(model, model_path)

if __name__ == '__main__':
    best_x, best_y = train()
    #best_x = None
    #best_y = None
    test_prediction, test_y, test_dates, test_base_prices = test(best_x, best_y)
    save(test_prediction, test_y, test_dates, test_base_prices)