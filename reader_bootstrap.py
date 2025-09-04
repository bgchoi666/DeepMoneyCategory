# Copyright 2018 Bimghi Choi. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import random as rd
import statistics as stat
import path_bootstrap as path

def norm_data(config, data_path=None):
  """
  read normalized data

  kospi200 futures 5, 20, 65days-forward  predictions text files with 943 input features,
  and performs mini-batching of the inputs.

  Args:
    config : configuration class
    data_path: not use

  Returns:
    tuple (train_data, valid_data, test_data)
    where each of the data objects can be passed to producer function.
  """

  raw_df = pd.read_csv(path.file_name, encoding="ISO-8859-1")
  raw_df = raw_df[:-config.predict_term]

  # column 0 : date, train, test가 시작되는(끝나는) index 계산
  train_start_index = len(raw_df[raw_df['date'] <= config.train_start]) - 1
  train_actual_start = max(train_start_index - config.step_interval * (config.num_steps - 1), 0)
  test_start_index = len(raw_df[raw_df['date'] <= config.test_start]) - 1
  test_end_index = min(len(raw_df[raw_df['date'] < config.test_end]) - 1, # + config.predict_term,
                       len(raw_df['date']) -1)

  input_target_list = list(range(1, 1 + config.input_size))
  #input_target_list = [1, 934, 935, 936, 937]

  if config.predict_term == 1: input_target_list.append(config.input_size+1)
  if config.predict_term == 5: input_target_list.append(config.input_size+2)
  if config.predict_term == 20: input_target_list.append(config.input_size+3)
  if config.predict_term == 65: input_target_list.append(config.input_size+4)

  #train 데이터 생성
  train_data =  raw_df.values[train_actual_start: test_start_index, input_target_list]

  #test data 생성
  test_data = raw_df.values[test_start_index - config.step_interval * (config.num_steps-1) : test_end_index + 1, input_target_list]
  predict_data = test_data

  return train_data, test_data, predict_data, test_start_index

# get base date, today, today index, standard deviation for the test period
def make_index_date(test_data, config):

  test_data_size = len(test_data) - config.step_interval * (config.num_steps-1)

  index_df = pd.read_csv("index-ma-std-" + path.market + ".csv", encoding = "ISO-8859-1")
  test_start_index = len(index_df[index_df['date'] <= config.test_start]) - 1

  #예측 시점의 지수 list
  index = index_df.values[test_start_index: test_start_index + test_data_size, 1]
  #예측 대상일 list
  date = index_df.values[test_start_index + config.predict_term: test_start_index +  config.predict_term + test_data_size, 0]
  #trade_date data
  #예측하는 날짜 list
  z = index_df.values[test_start_index : test_start_index + test_data_size, 0]
  z = list(z)
  #예측 대상일 standard deviation
  std = index_df.values[test_start_index + config.predict_term: test_start_index +  config.predict_term + test_data_size, 7]

  return index, date, z, std

# create time-series data for LSTM input
def producer(raw_data, num_steps, step_interval, input_size, output_size, mode, name=None):
  """produce time-series data.

  This chunks up raw_data into series of examples and returns Tensors that
  are drawn from these series.

  Args:
    raw_data: one of the raw data outputs from futures data.
    num_steps: int, the number of unrolls.
    step_interval : days between steps
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the target data(65days-forward kospi200 futures values).

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """

  # nput과 target 분리하여 반환
  dataX, dataY = [], []

  # series 데이터 건수 계산
  size = len(raw_data) - (num_steps-1) * step_interval

  for i in range(size):
    input_list = list(range(i, i + num_steps * step_interval, step_interval))
    a = np.float32(raw_data[input_list, :input_size])
    b = np.float32(raw_data[input_list, input_size])

    # mode가 'train'일 때 random number에 따라 데이터 선택
    #if mode == 'train' and rd.random() < 0.2:
    #  for j in range(min(5, size  - i)):
    #    dataX.append(a)
    #    dataY.append(b)
    #    i += 1
    #if mode == 'test':
    dataX.append(a)
    dataY.append(b)

  # 3차원 array로 dimension 조정
  x = np.array(dataX).reshape(-1, num_steps, input_size)
  y = np.array(dataY).reshape(-1, num_steps, output_size)

  return x, y
