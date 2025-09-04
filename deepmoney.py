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

import reader
import path
import model
from datetime import datetime
import time
import shutil
import os
import random as rd
import logging
from datetime import datetime
import threading

# from tensorflow.models.rnn.ptb import reader

logging = tf.logging

BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"
GRU = "gru"

#log file setup


tf.logging.set_verbosity(tf.logging.INFO)

def data_type():
    return tf.float32

def calculate_recall_precision(real, prediction, today, profits):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(0, len(today)):
        if prediction[i] - today[i] > 0 and profits[i] != 0:
            if real[i] - today[i] > 0:
                true_positives += 1
            else:
                false_positives += 1
        elif profits[i] != 0:
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
    pi_del = 0
    for i in range(len(pred)):
        if profits[i] > 0 and pred[i] < real[i] + 1.64*std[i] and pred[i] > real[i] - 1.64*std[i]:
            piacc += 1
        elif profits[i] == 0: pi_del += 1
    piacc = piacc / (len(pred) - pi_del)
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

# set the total result variables
today_tot = []
date_tot = []
index_pred_tot = [[], [], [], []]
index_real_tot = []
index_today_tot = []
profits_tot = [[], [], [], []]
pred_values_tot = [[], [], [], []]
target_values_tot = []
std_tot = []

def gradual_train(k, M2M, config):

    # set the total result variables
    global today_tot
    global date_tot
    global index_pred_tot
    global index_real_tot
    global index_today_tot
    global profits_tot
    global pred_values_tot
    global target_values_tot
    global std_tot

    for i in range(len(config.grad_train_terms) - 2):
        if i > 0:  config.iter_steps = 300

        config.train_start = config.grad_train_terms[i]
        config.test_start = config.grad_train_terms[i + 1]
        config.test_end = config.grad_train_terms[i + 2]

        #create datasets from the given file
        dataset = reader.norm_data(config)
        train_data, test_data, predict_data, test_start_index = dataset

        #log.info(datetime.now().strftime("%Y %m %d %H %M %S"))
        #log.info("====================== train start ===========================")

        # train the model
        model.train(M2M, train_data, config)

        # Predict the model and recieve the results
        today, date, index_pred, index_real, index_today, profits, pred_values, target_values, std = model.predict(M2M, predict_data, config, test_start_index)

        # add the prediction results to the total variables
        if (k == 0):
            today_tot += today
            date_tot += list(date)
            index_pred_tot[k] += index_pred
            index_real_tot += index_real
            index_today_tot += list(index_today)
            profits_tot[k] += profits
            pred_values_tot[k] += pred_values
            target_values_tot += target_values
            std_tot += list(std)
        else:
            index_pred_tot[k] += index_pred            
            profits_tot[k] += profits
            pred_values_tot[k] += pred_values        

def main(_):

    #log file setup
    #log = set_log(path.log_dir)

    # Train and Predict gradually according to the list of grad_train_terms
    # each gradual_train repeated for 4 different combinations of step_interval and num_steps
    t = [ '' for i in range(4)]
    config = [ path.Config() for i in range(4)] #get configuration class as many as the number of threads
    M2M = [ '' for i in range(4)] # model poiter as many as the number of threads
    for j in range(4):
        config[j].num_steps = config[j].num_steps_list[j]
        config[j].step_interval = config[j].step_interval_list[j]

        # create many-to-many estimator
        M2M[j] = model.create_estimator(config[j])

        # create and run threads for 4 combinations
        t[j] = threading.Thread(target=gradual_train, args=(j, M2M[j], config[j]))
        t[j].start()

        #log.info(datetime.now().strftime("%Y %m %d %H %M %S"))
        #log.info("......... estimator creted ...........")

        #gradual_train(j, M2M, config)

    # wait until all threads end
    for i in range(4): t[i].join()



    global today_tot
    global date_tot
    global index_pred_tot
    global index_real_tot
    global index_today_tot
    global profits_tot
    global pred_values_tot
    global target_values_tot
    global std_tot

    # average the 4 training results
    index_pred_avg = [0 for i in range(len(index_pred_tot[0]))]
    profits_avg = [0 for i in range(len(index_pred_tot[0]))]
    pred_values_avg = [0 for i in range(len(index_pred_tot[0]))]
    for i in range(0, 4):
        for j in range(len(index_pred_avg)):
            index_pred_avg[j] += index_pred_tot[i][j]/4
            profits_avg[j] += profits_tot[i][j]/4
            pred_values_avg[j] += pred_values_tot[i][j]/4

    # write the total results to a file
    comp_results = {"date_base": today_tot, "real": target_values_tot, "prediction": pred_values_avg, "index_today": index_today_tot,
                    "index_real": index_real_tot, "index_pred": index_pred_avg, "date_pred": date_tot, "loss_profits": profits_avg,
                    "std": std_tot}
    # result direcctory가 없으면 새로 생성
    if not os.path.isdir(path.result_dir) :
        os.makedirs(path.result_dir)
    result_file = path.result_dir + "/" + path.market + "_" + "alpha" + str(config.alpha) + "beta" + str(config.beta) + \
                  "_" + str(config.predict_term) + "_ensemble_" +  \
                  str(config.rnn_mode) + "_" + str(config.hidden_size) + "_" + \
                  str(config.batch_size) + "_" + datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H-%M-%S') + ".csv"
    pd.DataFrame.from_dict(comp_results).to_csv(result_file, index=False)

    accuracy, _, _, _ = calculate_recall_precision(index_real_tot, index_pred_avg, index_today_tot, profits_avg)
    piacc = cal_piacc(std_tot, index_pred_avg, index_real_tot, profits_avg)

    # append pure accuracy, inside-band accuracy to the result file
    r = open(result_file, 'a')
    r.write("accuracy, acc with interval\n" + str(accuracy) + "," + str(piacc))

    #close_log(log)

if __name__ == "__main__":
    tf.app.run()
