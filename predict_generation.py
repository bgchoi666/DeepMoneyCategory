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

 $ python predict.py --data_path=C:/Users/Admin/Desktop/DeepMoney_v2.0

 target : rate
          1, 5, 20, 65 days after :
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd

import reader_generation as reader
import path_generation as path
from datetime import datetime
import time
import shutil
import os
import random as rd
import logging

flags = tf.flags

flags.DEFINE_string("data_path", "C:/Users/Admin/Desktop/Demian-marketbreaker",
                    "Where the training/test data is stored.")
flags.DEFINE_string("rnn_mode", "BASIC",
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"
GRU = "gru"

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
    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(num_layers)], state_is_tuple=True)

    # state 초기화
    state = cell.zero_state(batch_size, data_type())

    inputs = features["x"]

    outputs = []
    # step 수 만큼 반복해서 output을 생성한다. input의 shape은 inputs[batch_size, num_steps, input_size]
    # 매 step마다 ouput을 append하여 다시 reshape한다. output[batch_size, num_steps, output_size]
    with tf.variable_scope("RNN"):
         for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
                #for i in range(batch_size):
                #    for j in range(input_size):
                #        inputs[i, time_step, j].assign(out[i, j])
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            #else:
            #    (cell_output, state) = cell(inputs[:, 0, :], state)
            #out = tf.layers.dense(cell_output, units=input_size, activation=tf.nn.tanh, reuse=tf.AUTO_REUSE, name="dense2")
            #out1 = tf.layers.dense(cell_output, units=5, activation=tf.nn.tanh, reuse=tf.AUTO_REUSE, name="dense1")
            #outputs = tf.layers.dense(cell_output, units=output_size, activation=None, reuse=tf.AUTO_REUSE, name="dense2")
            outputs.append(cell_output)
         #output, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)


    # many-to-one option... 마지막 step의 값만 필요
    #output = tf.reshape(outputs[num_steps-1], [-1, hidden_size])
    # append의 결과 shape이 (num_steps, batch_size, hidden_size)가 되므로
    # 이것을 ( batch_size, num_steps, hidden_size)로 바꾼 후, 다시 (batch_size*num_steps, hidden_size)로....
    output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])

    #모든 batch, step의 결과(batch_size*step_numbers)를 입력 데이터로 하여 dense layer들이 처리한다.
    #1 dense hidden layer and 1 output layer
    #net = tf.layers.dense(output, units=100, activation=tf.nn.relu)
    #net2 = tf.layers.dense(net, units=10, activation=tf.nn.relu)
    logits = tf.layers.dense(output, units=output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))

    logit = tf.reshape(logits, [-1])
    """
    # tensorboard에 weight의 변화값 저장
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
    #target = tf.reshape(tf.slice(labels, [0, num_steps - 1, 0], [batch_size, 1, output_size]), [-1])
    target = tf.reshape(labels, [-1])

    # optimizer = tf.train.GradientDescentOptimizer(self._lr)
    # self._train_op = optimizer.minimize(loss=loss, global_step=tf.contrib.framework.get_or_create_global_step())
    # loss는 target과의 차를 squared sum한 것의 평균으로 한다.
    loss = tf.losses.mean_squared_error(target, logit)

    #lstm regularization을 위한 cost 계산
    #lstm_trainable_weights = cell.trainable_weights
    #regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in lstm_trainable_weights])
    #loss = loss + regularization_cost

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate)
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            target, logit)
    }

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
    #새로운 모델 생성 모드용 모델 디렉토리 - 모델 디렉토리에 random number를 붙임
    model_dir = path.model_dir +  "/" + path.market + "_" +  str(config.predict_term) + "_" + \
                str(config.step_interval) + "_" + str(config.num_steps) + "_" + str(config.rnn_mode) + "_" + str(config.hidden_size) + "_" + str(config.batch_size) + \
                "_" + str(config.num_layers) + "layers" + str(rd.randint(1, 1000))
    #기존 모델 재 사용 모드용 디레토리
    model_dir_cont = path.model_dir +  "/" + path.market + "_" +  str(config.predict_term) + "_" + \
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
        return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=model_params)
    if os.path.isdir(model_dir_cont) : #기존 모델 재사용 모드
        return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir_cont, params=model_params)
    else :
        os.makedirs(model_dir_cont)
        return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir_cont, params=model_params)

def train(model, train_data, config):
    #raw data frame으로부터  inputs, targets data 생성한 후
    #model training

    #train input, target data 생성
    inputs, targets = reader.producer(train_data, config.num_steps, config.step_interval, config.input_size, config.output_size, config.target_index)

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
    inputs, targets = reader.producer(test_data, config.num_steps, config.step_interval, config.input_size, config.output_size, config.target_index)

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
    return ev["loss"]

def predict(model, predict_data, config, test_start_index):
    # Print out predictions

    # test input, target data 생성
    inputs, targets = reader.producer(predict_data, config.num_steps, config.step_interval, config.input_size, config.output_size, config.target_index)

    # test input function setup
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": inputs},
        batch_size=1,
        num_epochs=1,
        shuffle=False)
    predictions = model.predict(input_fn=predict_input_fn)
    #target = np.reshape(targets[:, config.num_steps - 1, :], (-1, config.output_size))
    target = list(targets.reshape((len(targets) * config.num_steps)))

    index_today, date, today, std = reader.make_index_date(predict_data, config)

    pred_values = []
    target_values = []
    for i, p in enumerate(predictions):
        if i % config.num_steps == config.num_steps - 1:
            pred_values.append(p["results"])
            target_values.append(target[i])

    #pred_values = np.reshape(pred_values, (-1, config.output_size))

    return today, date, pred_values, target_values

def calculate_recall_precision(real, prediction, today):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(0, len(today)):
        if prediction[i] - today[i] > 0:
            if real[i] - today[i] > 0:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if real[i] - today[i] < 0:
                true_negatives += 1
            else:
                false_negatives += 1

    precision = 0
    recall = 0
    f1_score = 0
    # a ratio of correctly predicted observation to the total observations
    accuracy = (true_positives + true_negatives) \
               / (true_positives + true_negatives + false_positives + false_negatives)

    # precision is "how useful the search results are"
    if true_positives + false_positives > 0 :  precision = true_positives / (true_positives + false_positives)
    # recall is "how complete the results are"
    if true_positives + false_negatives > 0 : recall = true_positives / (true_positives + false_negatives)

    if not (precision == 0 and recall == 0) : f1_score = 2 / ((1 / precision) + (1 / recall))

    return accuracy, precision, recall, f1_score

def cal_piacc(std, pred, real, profits):
    piacc = 0
    for i in range(len(pred)):
        if profits[i] > 0 and pred[i] < real[i] + 1.64*std[i] and pred[i] > real[i] - 1.64*std[i]:
            piacc += 1
    piacc = piacc / len(pred)
    return piacc

def set_log(log_file_name):

    # log file, stream handler setup
    logger = logging.getLogger('tensorflow')
    formatter = logging.Formatter('%(asctime)s > %(message)s', datefmt = '%Y-%m-%d %H: %M: %S')

    fh = logging.FileHandler(log_file_name)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def close_log(logger):
    handlers = logger.handlers
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    log = set_log(path.log_file_name)

    dt = datetime.now()
    log.info(dt.strftime("%Y %m %d %H %M %S"))
    log.info("================= program starts =================")

    #get configuration class
    config = path.Config()

    # data_pathe에 있는 train, test파일을 읽어  data frame 생성, raw_data() : raed non-normalized data
    #if config.read_mode == "raw" : dataset = reader.raw_data(config, FLAGS.data_path)
    #else : dataset = reader.norm_data(config, FLAGS.data_path)
    #train_data, test_data, predict_data, test_start_index = dataset

    # create many-to-many estimator
    M2M = create_estimator(config)

    # set the total result variables
    today_tot = []
    date_tot = []
    index_pred_tot = []
    index_real_tot = []
    index_today_tot = []
    profits_tot = []
    pred_values_tot = []
    target_values_tot = []
    std_tot = []

    # Train and Predict gradually according to the list of grad_train_terms
    for i in range(len(config.grad_train_terms) - 2):
        if i > 0:  config.iter_steps = 300
        config.train_start = config.grad_train_terms[i]
        config.test_start = config.grad_train_terms[i + 1]
        config.test_end = config.grad_train_terms[i + 2]

        #create datasets from the given file
        dataset = reader.norm_data(config, FLAGS.data_path)
        train_data, test_data, predict_data, test_start_index = dataset

        # train the model
        train(M2M, train_data, config)

        # Predict the model and recieve the results
        today, date, pred_values, target_values = predict(M2M, predict_data, config, test_start_index)

        # add the prediction results to the total variables
        today_tot += today
        date_tot += list(date)
        pred_values_tot += pred_values
        target_values_tot += target_values
    #pred_values_tot 일자순으로 재정열해서 list로
    #if len(config.grad_train_terms) - 2 > 1:
    #    gen_values = np.reshape(np.concatenate(pred_values_tot, axix=0), (len(today_tot), config.output_size))
    #else: gen_values = pred_values_tot[0]

    # write the total results to a file
    comp_results = {"date_base": today_tot, "pred": pred_values_tot, "target": target_values_tot}
    #for i in range(config.output_size):
    #        comp_results["gen_var" + str(i)] = gen_values[:, i]

    # result direcctory가 없으면 새로 생성
    if not os.path.isdir(path.result_dir) :
        os.makedirs(path.result_dir)
    result_file = path.result_dir + "/" + path.market + "_" + \
                  str(config.predict_term) + "_" + str(config.step_interval) + "_" + str(config.num_steps) + "_" + \
                  str(config.rnn_mode) + "_" + str(config.hidden_size) + "_" + \
                  str(config.batch_size) + "_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H-%M-%S') + ".csv"
    pd.DataFrame.from_dict(comp_results).to_csv(result_file, index=False)

    dt = datetime.now()
    log.info(dt.strftime("%Y %m %d %H %M %S"))
    log.info("==================== program end =========================")
    close_log(log)

if __name__ == "__main__":
    tf.app.run()
