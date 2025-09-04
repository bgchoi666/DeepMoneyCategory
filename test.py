from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import datetime

if __name__ == "__main__":

    df = pd.read_csv("kospi200f_805.csv", encoding='euc-kr')
    dates = list(df['date'])
    forward_dates = []
    for i in range(len(dates)):
        forward_dates.append((datetime.datetime.strptime(dates[i], "%Y-%m-%d") + datetime.timedelta(days=30)).strftime("%Y-%m-%d"))
    print(forward_dates[-20:])
    #tf.keras.callbacks.EarlyStopping()

    #df = df.drop(1, axis=0)
    #df = df.reset_index(drop=True)
    #print(df.head(10))
    #data_y = df.pop('종가').values
    #dataset = tf.data.Dataset.from_tensor_slices((df.values, data_y)).batch(10)
    #print(next(iter(dataset))[1])


    #sigma_init = tf.random_normal_initializer()
    #sigma = abs(tf.Variable(sigma_init(shape=(1, 120, 1)), dtype='float32', name='sigma'))
    #print(sigma)
    #df = pd.read_csv("test.csv")
    #print(df[df['a'] < "2019-01-01"].index.max())

    #input_df = df.fillna(method='ffill')
    #input_df = (input_df - input_df.rolling(20).mean()) / input_df.rolling(20).std()
    #norm_df = pd.concat([input_df, df["b"]], axis=1)
    #norm_df.replace(np.nan, 0, inplace=True)
    #norm_df = norm_df.astype(float)
    #norm_df = norm_df.dropna(axis=0).reset_index(drop=True)
    #input_df.to_csv("test_result.csv", index=False)