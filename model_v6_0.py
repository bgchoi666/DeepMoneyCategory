# Copyright 2018 Bumghi Choi. All Rights Reserved.
#

"""Index predicting Many-to-Many LSTM model.

# the difference from v5.0
# predict today's open price from the previous close price
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
    - kospi200f_20norm_current.csv : for prediction after 1 week with 943 input features

 model functions are integrated into one
 config,alpha : the RRL applied rate
 config.beta : the kelly applied rate
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# imperative running
#tf.enable_eager_execution()

import pandas as pd

import reader_v6_0 as reader
import path_v6_0 as path
from datetime import datetime
import time
import shutil
import os
import random as rd
import logging
from datetime import datetime

tf.enable_eager_execution()

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
conversion = path.Config().conversion

def data_type():
    return tf.float32

def model_setup(config):
    model_params = {"learning_rate": config.learning_rate,
                    "rnn_mode": config.rnn_mode,
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers,
                    "batch_size": config.batch_size,
                    "num_steps": config.num_steps,
                    "output_size": config.output_size,
                    "input_size": config.input_size,
                    "init_scale": config.init_scale,
                    "predict_term": config.predict_term,
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "shuffle": config.shuffle}
    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=20 * 60,  # Save checkpoints every 20 minutes.
        keep_checkpoint_max=1,  # Retain the 10 most recent checkpoints.
    )
    return model_params, my_checkpointing_config

def model_fn(features, labels, mode, params):
    #customized many to many LSTM model function

    rnn_mode = params["config"].rnn_mode
    hidden_size1 = params["config"].hidden_size1
    hidden_size2 = params["config"].hidden_size2
    num_layers = params["config"].num_layers
    num_steps = params["config"].num_steps
    output_size = params["config"].output_size
    learning_rate = params["config"].learning_rate
    alpha = params["config"].alpha
    beta = params["config"].beta
    config = params["config"]

    shuffle = params["config"].shuffle
    batch_size = params["config"].batch_size

    if mode == tf.estimator.ModeKeys.PREDICT:
        batch_size = 1;

    # make one cell(LSTM layer) for ibdex prediction
    def make_cell():
        if rnn_mode == BASIC:
            #return tf.contrib.rnn.LSTMCell(
            #    hidden_size, use_peepholes=True, initializer=tf.contrib.layers.xavier_initializer(),
            #    forget_bias=0.0, state_is_tuple=True,
            #    reuse=False, activation=tf.nn.tanh)
            return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size1)
        if rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(
                hidden_size1, forget_bias=0.0, activation=tf.nn.tanh)
        if rnn_mode == GRU:
            #return tf.contrib.rnn.GRUCell(
            #    hidden_size, activation=tf.nn.tanh)
            return tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(hidden_size1)
        raise ValueError("rnn_mode %s not supported" % params["rnn_mode"])
        return cell

    # create LSTM layers as many as num_layer
    #endoder cell
    cell_1 = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(num_layers)], state_is_tuple=True)

    # state initialization
    state = cell_1.zero_state(batch_size, data_type())

    input_targets = features["x"]
    inputs = input_targets[:, :, :config.input_size]

    outputs = []
    # endocer: recurrently created steps, input shape is inputs[batch_size, num_steps, input_size]
    # the last state --> initial state of decoder
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


    # many-to-one option... need only the last step
    #logit = tf.reshape(logits[num_steps-1], [-1])

    # after appending, shape be (num_steps, batch_size, hidden_size)
    # reshaped to ( batch_size, num_steps, hidden_size), again to (batch_size*num_steps, hidden_size)....
    output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size1])


    #1 dense hidden layer and 1 output layer
    #net = tf.layers.dense(output, units=100, activation=tf.nn.relu)
    #net2 = tf.layers.dense(net, units=10, activation=tf.nn.relu)
    logits = tf.layers.dense(output, units=output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

    logit = tf.reshape(logits, [-1])

    base_targets = input_targets[:, :, config.input_size:config.input_size + 2]

    # produce real, predicted index values
    base_prices = tf.reshape(base_targets[:, :, 0], [-1])
    predict_prices = []
    for i in range(batch_size*num_steps):
        predict_prices.append(logit[i] + base_prices[i])
    predict_prices = tf.reshape(predict_prices, [batch_size, num_steps, 1])
    base_prices = tf.reshape(base_prices, [batch_size, num_steps, 1])
    target_prices = tf.reshape(base_targets[:, :, 1], [batch_size, num_steps, 1])
    diff = tf.subtract(predict_prices, base_prices)

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
        "results": logit,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # when only the last step of target data used ...
    #targets = tf.slice(targets, [0, num_steps - 1, 0], [batch_size, 1, 1])

    target = tf.reshape(labels, [-1])

    # optimizer = tf.train.GradientDescentOptimizer(self._lr)
    # self._train_op = optimizer.minimize(loss=loss, global_step=tf.contrib.framework.get_or_create_global_step())

    # loss is the average of squared sum of difference from target
    loss = tf.losses.mean_squared_error(target, logit)

    #the cost for lstm regularization
    #lstm_trainable_weights = cell.trainable_weights
    #regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in lstm_trainable_weights])
    #loss = loss + regularization_cost

    if config.alpha != 0:
        # add RRL cost - maximize downside sharp ratio
        r = [] # price difference
        for i in range(batch_size):
            r.append(0) # the first step has no difference
            for j in range(1, num_steps):
                r.append(target_prices[i][j][0] - target_prices[i][j-1][0])
        r = tf.reshape(r, [batch_size, num_steps, 1])

        # 1 if (pred - base) * (target - base) > 0, -1 otherwise
        F = [] # price difference
        for i in range(batch_size):
            for j in range(num_steps):
                F.append(tf.sign((target_prices[i][j][0] - base_prices[i][j][0])*(predict_prices[i][j][0] - base_prices[i][j][0])))
        F = tf.reshape(F, [batch_size, num_steps, 1])

        # calc returns from each step in batches
        R = []
        for i in range(batch_size):
            for j in range(num_steps):
                R.append(abs(target_prices[i, j, 0] - base_prices[i, j, 0]) * (F[i, j, 0] - 0.00003)/base_prices[i, j, 0])
        R = tf.reshape(R, [batch_size, num_steps, 1])

        # calc downside sharp ratio

        # downside returns
        DR = []
        for i in range(batch_size):
            for j in range(num_steps):
                DR.append(tf.minimum(0.0, R[i, j, 0]))
        DR = tf.reshape(DR, [batch_size, num_steps, 1])

        # calc. downside sharp ratio
        #s = []
        #for i in range(batch_size):
        #   std =  tf.keras.backend.std(DR[i, :, 0])
        #   s.append(tf.reduce_mean(R[i, :, 0])/tf.maximum(0.01, std))

        # calc. downside sharp ratio
        loss2 = tf.reduce_mean(R) / (tf.keras.backend.std(DR) + 0.001)
    else:
        loss2 = 0

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

    if config.beta != 0:
        #compute maximum drawdown

        #accm_profit = [0.0]
        #for i in range(batch_size):
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
                accm_profit[0] = tf.sign((predict_prices[0, num_steps - 1, 0] - base_prices[0, num_steps - 1, 0]) *
                                         (target_prices[0, num_steps-1, 0] - base_prices[0, num_steps-1, 0])) * \
                                 abs(target_prices[0, num_steps-1, 0] - base_prices[0, num_steps-1, 0])
            else:
                accm_profit[i] = accm_profit[i-1] + tf.sign((predict_prices[i, num_steps - 1, 0] - base_prices[i, num_steps - 1, 0]) *
                                                            (target_prices[i, num_steps-1, 0] - base_prices[i, num_steps-1, 0])) * \
                                                    abs(target_prices[i, num_steps-1, 0] - base_prices[i, num_steps-1, 0])
        loss3 = (tf.reduce_max(accm_profit) - tf.reduce_min(accm_profit))/batch_size
    else:
        loss3 = 0

    # minimize the loss of prediction network and maximize downside sharp ratio of RRL
    # and maximize the difference between average profit and loss
    loss = loss - config.alpha * loss2 + config.beta * loss3

    if mode == tf.estimator.ModeKeys.EVAL:
        # Calculate root mean squared error as additional eval metric
        #eval_metric_ops = {
        #    "rmse": tf.metrics.root_mean_squared_error(target, logit),
        #    #"accuracy": tf.metrics.accuracy(tf.sign(target), tf.sign(logit))
        #}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
        )#eval_metric_ops=eval_metric_ops)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)

def return_one():
    return 1
def return_zero():
    return 0

def create_estimator(config, dir):
    #configuration parameter --> model parameter
    model_params = {"config": config}

    # only the last checkpoint saved
    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=20 * 60,  # Save checkpoints every 20 minutes.
        keep_checkpoint_max=1,  # Retain the 10 most recent checkpoints.
    )

    if config.mode == "p":
        model_dir = path.model_dir + "/" + dir
        if os.path.isdir(model_dir):  # reuse existing model
            return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=model_params,
                                          config=my_checkpointing_config)
        else:
            print("model directory path not exist!")
            exit(1)

    if dir == "":
        # model directory for new creation mode  - random number added to the end of directory
        #model_dir = path.model_dir + "/" + path.market + "_" + str(config.predict_term) + "_" + \
        #            str(config.step_interval) + "_" + str(config.num_steps) + "_" + \
        #            str(config.hidden_size1) + "_" + str(config.batch_size) + "_" + datetime.fromtimestamp(
        #    time.time()).strftime('%Y%m%d%H%M%S')
        # model directory for new creation mode  - random number added to the end of directory

        if config.rmode:
            reinforce = "A"
        else:
            reinforce = "T"
        model_dir = path.model_dir + "/" + path.market + "_" + str(config.predict_term) + "day_" + \
                    str(config.alpha) + "_" + str(config.beta) + "_" + \
                    reinforce + "_" + datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')

        try:
            shutil.rmtree(model_dir)
        except OSError as e:
            if e.errno == 2:
                # not existing file or directory
                pass
            else:
                raise
        return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=model_params, config=my_checkpointing_config)

    if os.path.isdir(dir) : # reuse existing model
        return tf.estimator.Estimator(model_fn=model_fn, model_dir=dir, params=model_params, config=my_checkpointing_config)
    else:
        print("model directory not exist")
        exit(1)


def train(model, train_data, config):
    #model training

    #train input, target data
    inputs, targets = reader.producer(train_data, config.num_steps, config.step_interval, config.input_size, config.output_size)

    # target values created asscording to conversion type
    # create target values according to given conversion type
    target = []
    for i in range(len(targets)):
        for step in range(config.num_steps):
            if conversion == 'diff': target.append(targets[i, step, 1] - targets[i, step, 0])
            if conversion == 'rate': target.append(
                (targets[i, step, 1] - targets[i, step, 0]) / targets[i, step, 0] * 100)
            if conversion == 'norm': target.append((targets[i, step, 0] - 67) / 30)
            if conversion == 'sign' : target.append(np.sign(targets[i, step, 1] - targets[i, step, 0]))
    target_values = np.array(target).reshape(-1, config.num_steps, 1)

    #train input function setup
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(inputs)},
        y=np.array(target_values),
        batch_size=config.batch_size,
        num_epochs=None,
        shuffle=config.shuffle)

    model.train(input_fn=train_input_fn, steps=config.iter_steps)

def test(model, test_data, config):
    # Score accuracy

    # testinput, target data
    inputs, targets = reader.producer(test_data, config.num_steps, config.step_interval, config.input_size, config.output_size)

    # target values created asscording to conversion type
    # create target values according to given conversion type
    target = []
    for i in range(len(targets)):
        for step in range(config.num_steps):
            if conversion == 'diff': target.append(targets[i, step, 1] - targets[i, step, 0])
            if conversion == 'rate': target.append(
                (targets[i, step, 1] - targets[i, step, 0]) / targets[i, step, 0] * 100)
            if conversion == 'norm': target.append((targets[i, step, 0] - 67) / 30)
            if conversion == 'sign' : target.append(np.sign(targets[i, step, 1] - targets[i, step, 0]))
    target_values = np.array(target).reshape(-1, config.num_steps, 1)

    # test input function setup
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(inputs)},
        y=np.array(target_values),
        batch_size=config.batch_size,
        num_epochs=1,
        shuffle=False)

    ev = model.evaluate(input_fn=test_input_fn)
    print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error : %s" % ev["rmse"])
    print("accuracy : %s" % ev["accuracy"])
    return ev["loss"]

def predict(model, predict_data, config, test_start_index):
    # Print out predictions

    # test input, target data
    inputs, targets = reader.producer(predict_data, config.num_steps, config.step_interval, config.input_size, config.output_size)

    # target values created asscording to conversion type
    # create target values according to given conversion type
    target = []
    for i in range(len(targets)):
        for step in range(config.num_steps):
            if conversion == 'diff': target.append(targets[i, step, 1] - targets[i, step, 0])
            if conversion == 'rate': target.append(
                (targets[i, step, 1] - targets[i, step, 0]) / targets[i, step, 0] * 100)
            if conversion == 'norm': target.append((targets[i, step, 0] - 67) / 30)
            if conversion == 'sign' : target.append(np.sign(targets[i, step, 1] - targets[i, step, 0]))

    # test input function setup
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": inputs},
        batch_size=1,
        num_epochs=1,
        shuffle=False)

    predictions = model.predict(input_fn=predict_input_fn)

    index_close, index_open, date, today, std = reader.make_index_date(predict_data, config)

    pred_values = []
    target_values = []
    asset_rebal = pred_values
    for i, p in enumerate(predictions):
        if i % config.num_steps == config.num_steps - 1:
            pred_values.append(p["results"])
            target_values.append(target[i])
    #for j in range(len(pred_values)):
    #    if pred_values[j] > 0:
    #        asset_rebal.append(1)
    #    else:
    #        asset_rebal.append(-1)

    index_real = []
    index_pred = []
    profits = []

    # restore to original index values according to market, conversion type
    for i in range(len(index_close)):
        if target[i] != 'Nan':
            if config.conversion == 'rate':
                index_real.append(round(target_values[i]*index_close[i]/100 + index_close[i], 2))
                index_pred.append(pred_values[i]*index_close[i]/100 + index_close[i])
            if config.conversion == 'diff':
                index_real.append(round(target_values[i] + index_close[i], 2))
                index_pred.append(pred_values[i] + index_close[i])
            if config.conversion == 'norm':
                index_real.append(round(target_values[i]*30 + 67, 2))
                index_pred.append(pred_values[i]*30 + 67)
            if config.conversion == 'sign' :
                index_real.append(round(targets[i, config.num_steps - 1, 1], 2))
                index_pred.append(index_close[i] * (pred_values[i]*0.1 + 1))

            if i > 0 and index_close[i] == index_close[i-1]:
                profits.append(0)
            else:
                if config.conversion == 'sign' :
                    if pred_values[i] > 0:
                        profits.append(index_real[i] - index_close[i])
                    else:
                        profits.append(index_close[i] - index_real[i])
                else:
                    if (index_close[i] - index_real[i])*(index_close[i] - index_pred[i]) > 0:
                        profits.append(abs(index_close[i] - index_real[i]))
                    else:
                        profits.append(-abs(index_close[i] - index_real[i]))
        else:
            index_real.append('NaN')
            profits.append('Nan')
            if config.conversion == 'rate':
                index_pred.append(pred_values[i]*index_close[i]/100 + index_close[i])
            if config.conversion == 'diff':
                index_pred.append(pred_values[i] + index_close[i])
            if config.conversion == 'norm':
                index_pred.append(pred_values[i]*30 + 67)
            if config.conversion == 'sign':
                index_pred.append(index_close[i] * (pred_values[i]*0.1 + 1))
    return today, date, index_pred, index_real, index_close, profits, pred_values, target_values, std, asset_rebal


