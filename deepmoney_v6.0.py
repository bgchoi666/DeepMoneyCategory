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
import model_v6_0 as model
from datetime import datetime, timedelta
import time
import shutil
import os
import random as rd
import logging
import threading
import sys
from multiprocessing import Process
import numpy as np

# from tensorflow.models.rnn.ptb import reader

BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"
GRU = "gru"

#log file setup

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags

flags.DEFINE_bool(
    "ensemble", False,
    "True if an ensemble mode of 4 different hyper-para sets is set")
flags.DEFINE_bool(
    "gradual", False,
    "True if gradual training by 1 month is set")
FLAGS = flags.FLAGS

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
    # recall is "how complete the negative results are"
    if true_positives + false_negatives > 0 : recall = true_negatives / (true_negatives + false_positives)

    if precision != 0 and recall != 0 : f1_score = 2 / ((1 / precision) + (1 / recall))

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

def no_gradual_train(k, M2M, config):

    # log file setup
    log = set_log(path.log_dir)

    log.info(datetime.now().strftime("%Y %m %d %H %M %S"))
    log.info("====================== no gradual train start ===========================")

    #create datasets from the given file
    rf, config = reader.norm_data(config)
    dataset = reader.get_dataset(rf, config)

    train_data1, train_data2, test_data, predict_data, test_start_index = dataset

    # train and evaluate the model every iter_steps steps
    #i = 0
    #while (i < config.iter_steps_max):
    #    model.train(M2M, train_data, config)
    #    model.test(M2M, test_data, config)
    #    i += config.iter_steps

    #train without intermediate evaluation
    config.iter_steps = config.iter_steps_max

    RRL = config.alpha
    kelly = config.beta
    batch_size = config.batch_size
    shuffle = config.shuffle

    if config.train_start1 < config.train_end1 and config.mode != 'p': model.train(M2M, train_data1, config)
    if config.train_start2 < config.train_end2 and config.mode != 'p': model.train(M2M, train_data2, config)
    if config.test_start >= config.test_end:
        print("wrong test term!!!!")
        exit(-1)

    if kelly != 0 and config.rmode and config.mode !='p':
        config.alpha = 0
        config.beta = kelly
        config.shuffle = True
        config.batch_size = 100
        M2M = model.create_estimator(config, M2M.model_dir)
        if config.train_start1 < config.train_end1: model.train(M2M, train_data1, config)
        if config.train_start2 < config.train_end2: model.train(M2M, train_data2, config)

    log.info(datetime.now().strftime("%Y %m %d %H %M %S"))
    log.info("====================== no gradual train end ===========================")

    #restore the config
    config.alpha = RRL
    config.kelly = kelly
    config.batch_size = batch_size
    config.shuffle = shuffle

    # Predict the model and recieve the results
    today, date, index_pred, index_real, index_base, profits, pred_values, target_values, std, rebal = model.predict(M2M, predict_data, config, test_start_index)

    # save the prediction results to the temporary file
    # assign index k to each hyper parameter setting in ensemble mode
    tmp_results = {"date_base": today, "date_pred": date,
                     "iiindex_base": index_base, "iindex_real": index_real,"index_pred_" + str(k): index_pred,
                    "loss_profits_" + str(k): profits, "asset_rebalancing" + str(k): rebal, "std": std}
    pd.DataFrame.from_dict(tmp_results).to_csv("ensemble" + str(k) + path.tmp_file_name, index=False)

def gradual_train(k, M2M, config):

    # log file setup
    log = set_log(path.log_dir)

    log.info(datetime.now().strftime("%Y %m %d %H %M %S"))
    log.info("====================== gradual train start #" + str(k) + " ===========================")

    first_gradual = True
    rf = []

    # save configuration
    RRL = config.alpha
    kelly = config.beta
    batch_size = config.batch_size
    shuffle = config.shuffle
    while config.train_end1_index >= config.test_end_index:

        # train and evaluate the model every iter_steps
        #n = 0
        #while n < config.iter_steps_max:
        #    model.train(M2M, train_data, config)
        #    model.test(M2M, test_data, config)
        #    n += config.iter_steps

        # if the first gradual step, set the indices of train. test dataset
        # else use gradually extended indices
        if first_gradual:
            #create datasets from the given file
            rf, config = reader.norm_data(config)
            train_data1, train_data2, test_data, predict_data, test_start_index = reader.get_dataset(rf, config)
            if config.mode != 'p':
                config.iter_steps = 1
                model.train(M2M, train_data2, config)
                config.iter_steps = config.iter_steps_max
        else:
            config.iter_steps = config.gradual_steps
            train_data1, train_data2, test_data, predict_data, test_start_index = reader.get_dataset(rf, config)

        log.info("train: " + str(config.train_start2) + "~" + str(config.train_end2))
        log.info("test: " + str(config.test_start) + "~" + str(config.test_end))

        if config.mode != 'p':

            if config.rmode:
                if RRL != 0:
                    if kelly != 0:
                        config.alpha = RRL
                        config.beta = 0
                        config.shuffle = shuffle
                        config.batch_size = batch_size
                        M2M = model.create_estimator(config, M2M.model_dir)
                    model.train(M2M, train_data2, config)
                if kelly != 0:
                    if RRL != 0:
                        config.alpha = 0
                        config.beta = kelly
                        config.shuffle = False
                        config.batch_size = 100
                        M2M = model.create_estimator(config, M2M.model_dir)
                    model.train(M2M, train_data2, config)
                if RRL == 0 and kelly == 0:
                    model.train(M2M, train_data2, config)
            else:
                model.train(M2M, train_data2, config)

        # Predict the model and recieve the results
        if len(predict_data) != 0:
            today, date, index_pred, index_real, index_base, profits, pred_values, target_values, std, rebal = model.predict(M2M, predict_data, config, test_start_index)

        # if not the first gadual sequence, load already saved temporary file
        if not first_gradual and os.path.isfile("ensemble" + str(k) + path.tmp_file_name):
            # read the saved temporary file
            tmp = pd.read_csv("ensemble" + str(k) + path.tmp_file_name, encoding="ISO-8859-1")

            # concate newly produced results
            today = list(tmp["date_base"].values) + today
            date = list(tmp["date_pred"].values) + date
            index_base = list(tmp["iiindex_base"].values) + index_base
            index_real = list(tmp["iindex_real"].values) + index_real
            index_pred = list(tmp["index_pred_" + str(k)].values) + index_pred
            profits = list(tmp["loss_profits_" + str(k)].values) + profits
            std = list(tmp["std"]) + std
            rebal = list(tmp["asset_rebalancing_" + str(k)].values) + rebal
        if len(predict_data) != 0:
            # save the concated results to temporary file
            tmp_results = {"date_base": today, "date_pred": date,
                           "iiindex_base": index_base, "iindex_real": index_real, "index_pred_" + str(k): index_pred,
                           "loss_profits_" + str(k): profits, "asset_rebalancing_" + str(k): rebal, "std": std}
            pd.DataFrame.from_dict(tmp_results).to_csv("ensemble" + str(k) + path.tmp_file_name, index=False)

            accuracy, _, down_accuracy, _ =  calculate_recall_precision(index_real, index_pred, index_base, profits)
            piacc = cal_piacc(std, index_pred, index_real, profits)

            # inermediate accuracy, average, max loss/profit written to log file
            log.info("accuracy: " + str(accuracy) + ", down accuracy: "  + str(down_accuracy) + ", acc with interval: " + str(piacc) + "\n")
            log.info("\n avg_profit, avg_loss, max_profit, max_loss\n" +
                   str(positive_avg_profits(profits)) + "," + str(negative_avg_profits(profits)) + "," +
                   str(max(list(profits))) + "," + str(min(list(profits))))

        # set up the next gradual step
        if config.train_end2_index - config.train_start2_index > 1000:
            config.train_start2_index += config.gradual_term
        config.train_end2_index = config.test_end_index
        config.test_start_index = config.train_end2_index + 1
        config.test_end_index = config.test_start_index + config.gradual_term

        if config.test_start_index < config.train_end1_index and  config.test_end_index > config.train_end1_index:
            config.test_end_index = config.train_end1_index

        first_gradual = False

    #restore the config
    config.alpha = RRL
    config.kelly = kelly
    config.batch_size = batch_size
    config.shuffle = shuffle

    log.info(datetime.now().strftime("%Y %m %d %H %M %S"))
    log.info("====================== gradual train end #" + str(k) + " ===========================")

def main(_):

    #options fetched from arguments
    #ensemble = tf.bool(sys.argv[1])
    #gradual = tf.bool(sys.argv[2])

    # log file setup
    log = set_log(path.log_dir)

    #options fetched from configuaration
    ensemble = path.Config().ensemble
    gradual = path.Config().gradual

    log.info("train term 1:" + (path.Config()).train_start1 + "~" + (path.Config()).train_end1)
    log.info("train term 2:" + (path.Config()).train_start2 + "~" + (path.Config()).train_end2)

    if ensemble:

       log.info(datetime.now().strftime("%Y %m %d %H %M %S"))
       log.info("====================== ensemble train start ===========================")

       tot_num_thread = path.num_threads
       # Train and Predict gradually according to the list of grad_train_terms
       # each gradual_train repeated for 4 different combinations of step_interval and num_steps
       t = [ '' for i in range(tot_num_thread)]
       config = [path.Config() for i in range(tot_num_thread)] #get configuration class as many as the number of threads
       M2M = [ '' for i in range(tot_num_thread)] # model pointer as many as the number of threads
       for j in range(tot_num_thread):
           config[j].num_steps = config[j].num_steps_list[j]
           config[j].step_interval = config[j].step_interval_list[j]

           # create many-to-many estimator
           if config.mode != 'p' and config.reset == True:
               M2M[j] = model.create_estimator(config[j], "")
           else:
               M2M[j] = model.create_estimator(config[j], path.model_dirs[j])
           log.info("mode; created, " + str(M2M[j].model_dir))

           # create and run threads for 4 combinations
           if gradual:
               #t[j] = threading.Thread(target=gradual_train, args=(j, M2M[j], config[j]))#, log))
               t[j] = Process(target=gradual_train, args=(j, M2M[j], config[j]))
           else: 
               #t[j] = threading.Thread(target=no_gradual_train, args=(j, M2M[j], config[j]))
               t[j] = Process(target=no_gradual_train, args=(j, M2M[j], config[j]))
           t[j].start()

       # wait until all threads end
       for i in range(tot_num_thread): t[i].join()

       log.info(datetime.now().strftime("%Y %m %d %H %M %S"))
       log.info("====================== ensemble train end, saving the results ===========================")

       # read the temporarily saved result files as many as the number of threads
       res = [pd.read_csv("ensemble" + str(i) + path.tmp_file_name, encoding="ISO-8859-1") for i in range(path.num_threads)]

       # average predict values in ensemble instances
       index_pred_avg  = [0 for k in range(len(res[0]["date_base"]))]
       profits_avg = [0 for k in range(len(res[0]["date_base"]))]
       for j in range(len(res[0]["date_base"])):
           for i in range(path.num_threads):
               index_pred_avg[j] += res[i]["index_pred_" + str(i)].values[j] / path.num_threads
           if (res[0]["iiindex_base"].values[j]  - res[0]["iindex_real"].values[j]) * (res[0]["iiindex_base"].values[j]  - index_pred_avg[j]) > 0:
               profits_avg[j] = abs(res[0]["iiindex_base"].values[j]  - res[0]["iindex_real"].values[j])
           else:
               profits_avg[j] = -abs(res[0]["iiindex_base"].values[j] - res[0]["iindex_real"].values[j])
       res[0]["index_pred_avg"] = index_pred_avg
       res[0]["profits_avg"] = profits_avg

       if config[0].mode == 'p':
           result_dir = "predict/" + path.result_dir
       else:
           result_dir = path.result_dir
       # if not existing, create a new result direcctory
       if not os.path.isdir(result_dir) :
           os.makedirs(result_dir)
       result_file = result_dir + "/" + path.market + "_" + "alpha" + str(config[0].alpha) + "beta" + str(config[0].beta) + \
                     "_" + str(config[0].predict_term) + "day_" + "reinf_" + str(config[0].rmode) + "_" + \
                     datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H-%M-%S') + ".csv"
       # save the results to a file
       pd.DataFrame.from_dict(res[0]).to_csv(result_file, index=False)

       accuracy, _, down_accuracy, _ = calculate_recall_precision(res[0]["iindex_real"].values, res[0]["index_pred_avg"].values, res[0]["iiindex_base"].values, res[0]["profits_avg"].values)
       piacc = cal_piacc(res[0]["std"].values, res[0]["index_pred_avg"].values, res[0]["iindex_real"].values, res[0]["profits_avg"].values)

       average_profit = positive_avg_profits(res[0]["profit_avg"].values)
       average_loss = negative_avg_profits(res[0]["profit_avg"].values)
       list_index = list(res[0]["iiindex_base"].values)
       average_index = sum(list_index) / len(list_index)
       kelly_rate = average_index * (accuracy / -average_loss - (1 - accuracy) / average_profit)

       # append pure accuracy, inside-band accuracy to the result file
       r = open(result_file, 'a')
       r.write("accuracy, down accuracy, acc with interval, kelly rate\n" + str(accuracy) + "," + str(down_accuracy) + "," + str(piacc) + "," + str(kelly_rate))
       r.write("\n ensemble, gradual, RRL, kelly, layers, hidden_size, batch_size, iteration\n" +
               str(config[0].ensemble) + "," + str(config[0].gradual) + "-" + str(config[0].gradual_term) + "," +
               str(config.alpha) + "," + str(config.beta) + "," +
               str(config[0].num_layers) + "," + str(config[0].hidden_size1) + "," +
               str(config[0].batch_size) + "," + str(config[0].iter_steps_max) + "-" + str(config[0].gradual_steps))
       r.write("\n interval, time steps, avg_profit, avg_loss, avg_index, max_profit, max_loss, accuracy, down accuracy, acc with interval, kelly rate\n")
       for i in range(tot_num_thread):
           accuracy, _, down_accuracy, _ = calculate_recall_precision(res[i]["iindex_real"].values,
                                                                      res[i]["index_pred_" + str(i)].values,
                                                                      res[i]["iiindex_base"].values,
                                                                      res[i]["loss_profits_" + str(i)].values)
           piacc = cal_piacc(res[i]["std"].values, res[i]["index_pred_" + str(i)].values, res[i]["iindex_real"].values,
                             res[i]["loss_profits_" + str(i)].values)

           average_profit = positive_avg_profits(res[i]["loss_profits_" + str(i)].values)
           average_loss = negative_avg_profits(res[i]["loss_profits_" + str(i)].values)
           list_index = list(res[i]["iiindex_base"].values)
           average_index = sum(list_index) / len(list_index)
           kelly_rate = average_index * (accuracy / -average_loss - (1 - accuracy) / average_profit)

           r.write(
               str(config[i].step_interval) + "," + str(config[i].num_steps) + "," +
               str(average_profit) + "," + str(average_loss) + "," +
               str(average_index) + "," +
               str(max(list(res[i]["loss_profits_" + str(i)].values))) + "," + str(min(list(res[i]["loss_profits_" + str(i)].values))) + "," +
               str(accuracy) + "," + str(down_accuracy) + "," + str(piacc) + str(kelly_rate) + "\n")

    else:

       log.info(datetime.now().strftime("%Y %m %d %H %M %S"))
       log.info("======================no-ensemble train-predict start ===========================")

       config = path.Config()

       # create datasets from the given file
       #dataset = reader.norm_data(config)
       #train_data, test_data, predict_data, test_start_index = dataset
       #index_today, _, _, _ = reader.make_index_date(predict_data, config)

       if config.mode == 'p' or config.model_reset == False:
           M2M = model.create_estimator(config, path.model_dirs[0])
       else:
           M2M = model.create_estimator(config, "")
       log.info("model created: " + str(M2M.model_dir))

       if gradual:
           gradual_train(0, M2M, config)
       else:
           no_gradual_train(0, M2M, config)

       log.info(datetime.now().strftime("%Y %m %d %H %M %S"))
       log.info("======================no-ensemble train-predict end, saving the results ===========================")
    
       # read the temporarily saved result file
       res = pd.read_csv("ensemble0" + path.tmp_file_name, encoding="ISO-8859-1")
    
       if config.mode == 'p':
           result_dir = "predict/" + path.result_dir
       else:
           result_dir = path.result_dir
       # if not existing, create a new result direcctory
       if not os.path.isdir(result_dir) :
           os.makedirs(result_dir)
       result_file = result_dir + "/" + path.market + "_" + "alpha" + str(config.alpha) + "beta" + str(config.beta) + \
                     "_" + str(config.predict_term) + "day_" + "reinf_" + str(config.rmode) + "_" + \
                     datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H-%M-%S') + ".csv"
       os.rename("ensemble0" + path.tmp_file_name, result_file)

       accuracy, _, down_accuracy, _ =  calculate_recall_precision(res["iindex_real"].values, res["index_pred_0"].values, res["iiindex_base"].values, res["loss_profits_0"].values)
       piacc = cal_piacc(res["std"].values, res["index_pred_0"].values, res["iindex_real"].values, res["loss_profits_0"].values)

       avg_index = sum(list(res["iiindex_base"].values)) / len(list(res["iiindex_base"].values))
       avg_profit = positive_avg_profits(res["loss_profits_0"].values)
       avg_loss = negative_avg_profits(res["loss_profits_0"].values)
       kelly = -accuracy/(avg_loss/avg_index) - (1-accuracy)/(avg_profit/avg_index)

       # append pure accuracy, inside-band accuracy to the result file
       r = open(result_file, 'a')
       r.write("accuracy, down accuracy, acc with interval\n" + str(accuracy) + "," + str(down_accuracy) + "," +  str(piacc))
       r.write("\n ensemble, gradual, RRL, kelly, time steps, interval, layers, hidden_size, batch_size, iteration\n" +
               str(config.ensemble) + "," + str(config.gradual) + "-" + str(config.gradual_term) + "," +
               str(config.alpha) + "," + str(config.beta) + "," +
               str(config.num_steps) + "," + str(config.step_interval) + "," +
               str(config.num_layers) + "," + str(config.hidden_size1) + "," +
               str(config.batch_size) + "," + str(config.iter_steps_max) + "-" + str(config.gradual_steps))
       r.write("\n avg_profit, avg_loss,  avg_index, max_profit, max_loss, kelly\n" +
               str(avg_profit) + "," + str(avg_loss) + "," + str(avg_index) + "," +
               str(max(list(res["loss_profits_0"].values))) + "," + str(min(list(res["loss_profits_0"].values))) + "," +
               str(kelly))

    close_log(log)

def positive_avg_profits(profits):
    positive_list = []
    for i in range(len(profits)):
        if profits[i] > 0 : positive_list.append(profits[i])
    positive_profits_avg = np.average(positive_list)
    return positive_profits_avg

def negative_avg_profits(profits):
    negative_list = []
    for i in range(len(profits)):
        if profits[i] < 0 : negative_list.append(profits[i])
    negative_profits_avg = np.average(negative_list)
    return negative_profits_avg

if __name__ == "__main__":
    tf.app.run()
