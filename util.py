# Copyright 2020 Bumghi Choi. All Rights Reserved.
# util.py
# 각종 file path정보, configuration 정보, utility functions

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf
import logging

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
#config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)

# version_name
ver = "v8"
market = "kospi200f"

#data file directorty
read_dir = ""

#data file name
file_name = "kospi200f_805.csv"

#temporary file name
tmp_file_name = "tmp.csv"

#model directory
model_dir = "model_dir/" + ver

#result file directory
result_dir = "results_" + ver

#log file name
log_dir = "log/" + ver + ".log"

#the start & end date of the train & test from a model
train_start = "2000-01-03"
train_end = "2018-09-31"

test_start = "2019-01-01"
test_end = "2019-11-19"

remove_columns = []
target_column = '종가'

conversion = 'diff'
shuffle = True

SHUFFLE_BUF = 10000
BATCH_SIZE = 100

def read_datafile(file_path):
  dataframe = pd.read_csv(file_path, encoding='euc-kr')
  return dataframe

#the configuration of a model's hyperparameters
class Config(object):

    init_scale = 0.05
    learning_rate = 0.001

    predict_term = 2

    step_interval = 1
    num_steps = 20

    num_layers = 2
    hidden_size = [500, 500]
    batch_size = 20

    input_size = 809
    output_size = 1

    def __init__(self, df=None):
        if df is not None:
          for column in df.columns:
            if column == "init_scale": self.initial_scale = df[column].value[0] # a scale of initializing weights
            if column == "learning_rate": self.learning_rate = df[column].value[0]
            if column == "predict_term": self.predict_term = df[column].value[0]
            if column == "step_interval": self.step_interval = df[column].value[0]
            if column == "num_steps": self.num_steps = df[column].value[0]
            if column == "num_layers": self.num_layers = df[column].value[0]
            if column == "hidden_size":
              for i in range(self.num_layers):
                self.hidden_size[i] = df[column].value[i]
            if column == "batch_size": self.batch_size = df[column].value[0]
            if column == "input_size": self.input_size = df[column].value[0]
            if column == "output_size": self.output_size = df[column].value[0]
            if column == "alpha": self.alpha = df[column].value[0] #the coefficient of RRL loss
            if column == "beta": self.beta = df[column].value[0] #the coefficient of kelly loss
            if column == "conversion": self.conversion = df[column].value[0] #the type of target value conversion. 'rate', 'diff'

        # setting steps, intervals for each predict term in the ensemble mode
        if self.predict_term == 1:
          self.num_steps_list = [20, 20, 20, 20]
          self.step_interval_list = [1, 1, 1, 1]
        if self.predict_term == 2:
          self.num_steps_list = [20, 50, 20, 50]
          self.step_interval_list = [1, 1, 2, 2]
        if self.predict_term == 3:
          self.num_steps_list = [30, 50, 30, 20]
          self.step_interval_list = [1, 1, 2, 3]
        if self.predict_term == 5:
          self.num_steps_list = [50, 30, 20, 20]
          self.step_interval_list = [1, 2, 3, 5]
        if self.predict_term == 10:
          self.num_steps_list = [30, 20, 30, 20]
          self.step_interval_list = [1, 5, 5, 10]
        if self.predict_term == 20:
          self.num_steps_list = [50, 30, 20, 20]
          self.step_interval_list = [1, 5, 10, 20]
        if self.predict_term == 65:
          self.num_steps_list = [100, 20, 20, 20]
          self.step_interval_list = [1, 10, 20, 65]
        if self.predict_term == 130:
          self.num_steps_list = [100, 20, 20, 20]
          self.step_interval_list = [1, 10, 20, 65]

config = Config()

#configuration file을 읽어 configuration class 생성 후 반환
def read_configfile(file_path):
  df = pd.read_csv(file_path)
  config = Config(df)
  return config

#log file을 만들기 위한 logger생성
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

#accuracy, recall, precision 계산
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
    if true_positives + false_positives > 0:  precision = true_positives / (true_positives + false_positives)
    # recall is "how complete the negative results are"
    if true_positives + false_negatives > 0: recall = true_negatives / (true_negatives + false_positives)

    if precision != 0 and recall != 0: f1_score = 2 / ((1 / precision) + (1 / recall))

    return accuracy, precision, recall, f1_score

#예측값이 (실재값 - 1.64 * (실재값의 n일간 표준편차), 실재값 + 1.64 * (실재값의 n일간 표준편차))내에 존재하는지 여부 확인
#구간내에 들어오면서 방향성도 맞는 경우의 확률을 반환
def cal_piacc(std, pred, real, profits):
    piacc = 0
    pi_del = 0
    for i in range(len(pred)):
        if profits[i] > 0 and pred[i] < real[i] + 1.64*std[i] and pred[i] > real[i] - 1.64*std[i]:
            piacc += 1
        elif profits[i] == 0: pi_del += 1
    piacc = piacc / (len(pred) - pi_del)
    return piacc

#방향성 예측을 토대로 산정된 투자 이익 평균
def positive_avg_profits(profits):
    positive_list = []
    for i in range(len(profits)):
        if profits[i] > 0 : positive_list.append(profits[i])
    positive_profits_avg = np.average(positive_list)
    return positive_profits_avg

#방향성 예측을 토대로 산정된 투자 손실 평균
def negative_avg_profits(profits):
    negative_list = []
    for i in range(len(profits)):
        if profits[i] < 0 : negative_list.append(profits[i])
    negative_profits_avg = np.average(negative_list)
    return negative_profits_avg

def data_type():
    return tf.float32