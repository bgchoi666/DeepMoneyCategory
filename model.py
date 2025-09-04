# Copyright 2018 Bumghi Choi. All Rights Reserved.
#

"""Index predicting Many-to-Many LSTM model.

 The model described in:
 (Zaremba, et. al.) Recurrent Neural Network Regularization
 http://arxiv.org/abs/1409.2329
 has been modified for predicting kospi200 future index


 The hyperparameters used in the model:
 - init_scale - the initial scale of the weights
 - learning_rate - the initial value of the learning rate
 - num_layers - the number of LSTM layers
 - num_steps - the number of unrolled steps of LSTM
 - hidden_size - the number of LSTM units
 - batch_size - the batch size
 - input_size - the number of input nodes
 - file_name
 - conversion - the conversion type of target data: rate, difference, 20days norm
 - grad_train_term - [train_start_date1, train_end_date1(=test_start_date1=train_start_date2), train__end_date2(=test_end_date1=test_start_date2=train_start_date3), . . . ]

 The data required for this example is in the data/ dir of the
    - kosfi200f-943.csv : for prediction after 1 week with 943 input features

 To run:

 $ python sseepmoney.py --data_path=C:/Users/Admin/Desktop/DeepMoney_v2.0

 target : rate
          1, 5, 20, 65 days after :
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# imperative running
#tf.enable_eager_execution()

import pandas as pd

import reader
import path
from datetime import datetime
import time
import shutil
import os
import random as rd
import logging
from datetime import datetime

# from tensorflow.models.rnn.ptb import reader

#flags = tf.flags
logging = tf.logging

#flags.DEFINE_string("data_path", "DeepMoney_v2.0",
#                    "Where the training/test data is stored.")
#flags.DEFINE_string("rnn_mode", "BASIC",
#                    "The low level implementation of lstm cell: one of CUDNN, "
#                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
#                    "and lstm_block_cell classes.")
#FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"
GRU = "gru"

#log file setup


tf.logging.set_verbosity(tf.logging.INFO)

def data_type():
    return tf.float32


def model_fn(features, labels, mode, params):
    #customized many to many LSTM model function

    rnn_mode = params["rnn_mode"]
    hidden_size = params["hidden_size"]
    num_layers = params["num_layers"]
    batch_size = params["batch_size"]
    num_steps = params["num_steps"]
    input_size = params["input_size"]
    output_size = params["output_size"]
    learning_rate = params["learning_rate"]
    init_scale = params["init_scale"]
    predict_term = params["predict_term"]
    alpha = params["alpha"]
    beta = params["beta"]
    shuffle = params["shuffle"]

    if not (mode == tf.estimator.ModeKeys.TRAIN):
        batch_size = 1;

    # 한개의 cell(LSTM layer)을 만든다.
    def make_cell():
        if rnn_mode == BASIC:
            #return tf.contrib.rnn.LSTMCell(
            #    hidden_size, use_peepholes=True, initializer=tf.contrib.layers.xavier_initializer(),
            #    forget_bias=0.0, state_is_tuple=True,
            #    reuse=False, activation=tf.nn.tanh)
            return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size)
        if rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(
                hidden_size, forget_bias=0.0, activation=tf.nn.tanh)
        if rnn_mode == GRU:
            #return tf.contrib.rnn.GRUCell(
            #    hidden_size, activation=tf.nn.tanh)
            return tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(hidden_size)
        raise ValueError("rnn_mode %s not supported" % params["rnn_mode"])
        return cell

    # num_layers의 수 만큼 LSTM layer 생성
    #endoder cell
    cell_1 = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(num_layers)], state_is_tuple=True)
    #decoder cell
    cell_2 = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(num_layers)], state_is_tuple=True)

    # state 초기화
    state = cell_1.zero_state(batch_size, data_type())

    inputs = features["x"]

    outputs = []
    # endocer: step 수 만큼 반복해서 state 생성한다. input의 shape은 inputs[batch_size, num_steps, input_size]
    # 마지막 state를 decoding RNN의 초기 state로 입력한다.
    with tf.variable_scope("RNN1"):
         for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
                #for i in range(batch_size):
                #    for j in range(input_size):
                #        inputs[i, time_step, j].assign(out[i, j])
            (cell_output, state) = cell_1(inputs[:, time_step, :], state)
            #else:
            #    (cell_output, state) = cell(inputs[:, 0, :], state)
            #out = tf.layers.dense(cell_output, units=input_size, activation=tf.nn.tanh, reuse=tf.AUTO_REUSE, name="dense2")
            #out1 = tf.layers.dense(cell_output, units=5, activation=tf.nn.tanh, reuse=tf.AUTO_REUSE, name="dense1")
            #outputs = tf.layers.dense(cell_output, units=output_size, activation=None, reuse=tf.AUTO_REUSE, name="dense2")
            outputs.append(cell_output)
         #output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)


    # many-to-one option... 마지막 step의 값만 필요
    #logit = tf.reshape(logits[num_steps-1], [-1])
    # append의 결과 shape이 (num_steps, batch_size, hidden_size)가 되므로
    # 이것을 ( batch_size, num_steps, hidden_size)로 바꾼 후, 다시 (batch_size*num_steps, hidden_size)로....
    output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])

    """ 
    # decoder: step 수 만큼 반복해서 output을 생성한다. input은 encoder의 마지막 step의 output으로, shape은 [batch_size, hidden_size]
    # 매 step마다 ouput을 append하여 다시 reshape한다. output[batch_size*num_steps, hidden_size]
    with tf.variable_scope("RNN2"):
         for time_step in range(num_steps):
            #if time_step > 0: tf.get_variable_scope().reuse_variables()
                #for i in range(batch_size):
                #    for j in range(input_size):
                #        inputs[i, time_step, j].assign(out[i, j])
            (cell_output, state) = cell_2(cell_output, state)
            #else:
            #    (cell_output, state) = cell(inputs[:, 0, :], state)
            #out = tf.layers.dense(cell_output, units=input_size, activation=tf.nn.tanh, reuse=tf.AUTO_REUSE, name="dense2")
            #out1 = tf.layers.dense(cell_output, units=5, activation=tf.nn.tanh, reuse=tf.AUTO_REUSE, name="dense1")
            #outputs = tf.layers.dense(cell_output, units=output_size, activation=None, reuse=tf.AUTO_REUSE, name="dense2")
            outputs.append(cell_output)
         #output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)


    # many-to-one option... 마지막 step의 값만 필요
    #logit = tf.reshape(logits[num_steps-1], [-1])
    # append의 결과 shape이 (num_steps, batch_size, hidden_size)가 되므로
    # 이것을 ( batch_size, num_steps, hidden_size)로 바꾼 후, 다시 (batch_size*num_steps, hidden_size)로....
    output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
    """

    #모든 batch, step의 결과(batch_size*step_numbers)를 입력 데이터로 하여 dense layer들이 처리한다.
    #1 dense hidden layer and 1 output layer
    #net = tf.layers.dense(output, units=100, activation=tf.nn.relu)
    #net2 = tf.layers.dense(net, units=10, activation=tf.nn.relu)
    logits = tf.layers.dense(output, units=output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

    logit = tf.reshape(logits, [-1])
    """
    tv = tf.trainable_variables()
    first_layer = 0
    second_layer = 0
    dense_layer = 0

    for i in range(1443):
        first_layer += tv[0][i, 10]
    for i in range(1000):
        second_layer += tv[2][i, 10]
    for i in range(500):
        dense_layer += tv[4][i, 0]

    tf.summary.scalar("first_layer", first_layer)
    tf.summary.scalar("second_layer", second_layer)
    tf.summary.scalar("dense_layer", dense_layer)
    """
    #for i in range(400): tf.summary.scalar("second_layer_weight", tv[5][200,i])

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "results": logit
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # target data중 마지막 step에 있는 data만 사용할 떄...
    #targets = tf.slice(targets, [0, num_steps - 1, 0], [batch_size, 1, 1])
    target = tf.reshape(labels, [-1])

    # optimizer = tf.train.GradientDescentOptimizer(self._lr)
    # self._train_op = optimizer.minimize(loss=loss, global_step=tf.contrib.framework.get_or_create_global_step())
    # loss는 target과의 차를 squared sum한 것의 평균으로 한다.
    loss = tf.losses.mean_squared_error(target, logit)

    #lstm regularization을 위한 cost 계산
    #lstm_trainable_weights = cell.trainable_weights
    #regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in lstm_trainable_weights])
    #loss = loss + regularization_cost


    # add auxilary cost - the difference between the predictions of steps
    logit_pre = []
    for i in range(num_steps*batch_size):
        if shuffle :
            if i % num_steps != 0: logit_pre.append(logit[i-1])
            else: logit_pre.append(logit[i])
        else:
            if i > num_steps and i % num_steps == 0 : logit_pre.append(logit[i-num_steps])
            elif i == 0: logit_pre.append(logit[0])
            else: logit_pre.append(logit[i - 1])
    loss_aux1 = tf.losses.mean_squared_error(logit_pre, logit)

    # add auxilary cost - the difference of stnadard deviations between target and prediction
    loss_aux2 = (tf.keras.backend.std(target) - tf.keras.backend.std(logit))**2
    loss = loss + alpha*loss_aux1 + beta*loss_aux2

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(target, logit),
        "accuracy": tf.metrics.accuracy(tf.sign(target), tf.sign(logit))
    }

    if not (mode == tf.estimator.ModeKeys.TRAIN):
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

def create_estimator(config):
    #configuration에 있는 parameter 값들을 model parameter로 넘긴다.
    model_params = {"learning_rate": config.learning_rate,
                    "rnn_mode" : config.rnn_mode,
                    "hidden_size" : config.hidden_size,
                    "num_layers" : config.num_layers,
                    "batch_size" : config.batch_size,
                    "num_steps" : config.num_steps,
                    "output_size" : config.output_size,
                    "input_size" : config.input_size,
                    "init_scale" : config.init_scale,
                    "predict_term" : config.predict_term,
                    "alpha" : config.alpha,
                    "beta" : config.beta,
                    "shuffle" : config.shuffle}
    # 마지막 checkpoint만 저장
    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=20 * 60,  # Save checkpoints every 20 minutes.
        keep_checkpoint_max=1,  # Retain the 10 most recent checkpoints.
    )
    # #새로운 모델 생성 모드용 모델 디렉토리 - 모델 디렉토리에 random number를 붙임
    model_dir = path.model_dir +  "/" + path.market + "_" + str(config.predict_term) + "_" + \
                str(config.step_interval) + "_" + str(config.num_steps) + "_" + str(config.rnn_mode) + "_" + \
                str(config.hidden_size) + "_" + str(config.batch_size) + "_" + str(config.num_layers) + "layers" + str(rd.randint(1, 1000))
    #기존 모델 재 사용 모드용 디레토리
    model_dir_cont = path.model_dir +  "/" + path.market + "_" + "alpha" + str(config.alpha) + "beta" + str(config.beta) + "_" + str(config.predict_term) + "_" + \
                     str(config.step_interval) + "_" + str(config.num_steps) + "_" + str(config.rnn_mode) + "_" + str(config.hidden_size) + "_" + str(config.batch_size) + \
                     "_" + str(config.num_layers) + "layers"
    if config.model_reset == True: #새 모델 생성 모드
        try:
            shutil.rmtree(model_dir)
        except OSError as e:
            if e.errno == 2:
                # 파일이나 디렉토리가 없음!
                pass
            else:
                raise
        return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=model_params, config=my_checkpointing_config)
    if os.path.isdir(model_dir_cont) : #기존 모델 재사용 모드
        return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir_cont, params=model_params, config=my_checkpointing_config)
    else :
        os.makedirs(model_dir_cont)
        return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir_cont, params=model_params, config=my_checkpointing_config)

def train(model, train_data, config):
    #raw data frame으로부터  inputs, targets data 생성한 후
    #model training

    #train input, target data 생성
    inputs, targets = reader.producer(train_data, config.num_steps, config.step_interval, config.input_size, config.output_size)

    #train input function setup
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(inputs)},
        y=np.array(targets),
        batch_size=config.batch_size,
        num_epochs=None,
        shuffle=config.shuffle)

    model.train(input_fn=train_input_fn, steps=config.iter_steps)

def test(model, test_data, config):
    # Score accuracy

    # testinput, target data 생성
    inputs, targets = reader.producer(test_data, config.num_steps, config.step_interval, config.input_size, config.output_size)

    # test input function setup
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(inputs)},
        y=np.array(targets),
        batch_size=1,
        num_epochs=1,
        shuffle=False)

    ev = model.evaluate(input_fn=test_input_fn)
    print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error : %s" % ev["rmse"])
    print("accuracy : %s" % ev["accuracy"])
    return ev["loss"]

def predict(model, predict_data, config, test_start_index):
    # Print out predictions

    # test input, target data 생성
    inputs, targets = reader.producer(predict_data, config.num_steps, config.step_interval, config.input_size, config.output_size)

    # test input function setup
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": inputs},
        batch_size=1,
        num_epochs=1,
        shuffle=False)
    predictions = model.predict(input_fn=predict_input_fn)
    target = list(targets.reshape((len(targets)*config.num_steps)))

    index_today, date, today, std = reader.make_index_date(predict_data, config)

    pred_values = []
    target_values = []
    for i, p in enumerate(predictions):
        if i % config.num_steps == config.num_steps - 1:
            pred_values.append(p["results"])
            target_values.append(target[i])

    index_real = []
    index_pred = []
    profits = []

    #원 지수로 복원, 시장별, conversion type별 복원 방식이 다름
    for i in range(len(index_today)):
        if target[i] != 'Nan':
            if config.conversion == 'rate':
                index_real.append(target_values[i]*index_today[i]/100 + index_today[i])
                index_pred.append(pred_values[i]*index_today[i]/100 + index_today[i])
            if config.conversion == 'diff':
                index_real.append(target_values[i] + index_today[i])
                index_pred.append(pred_values[i] + index_today[i])
            if config.conversion == 'norm':
                index_real.append(target_values[i]*30 + 67)
                index_pred.append(pred_values[i]*30 + 67)
            if (index_today[i] - index_real[i])*(index_today[i] - index_pred[i]) > 0: profits.append(abs(index_today[i] - index_real[i]))
            else: profits.append(-abs(index_today[i] - index_real[i]))
        else:
            index_real.append('NaN')
            profits.append('Nan')
            if config.conversion == 'rate':
                index_pred.append(pred_values[i]*index_today[i]/100 + index_today[i])
            if config.conversion == 'diff':
                index_pred.append(pred_values[i] + index_today[i])
            if config.conversion == 'norm':
                index_pred.append(pred_values[i]*30 + 67)
    return today, date, index_pred, index_real, index_today, profits, pred_values, target_values, std


