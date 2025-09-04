# _*_ coding: utf-8 _*_

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
import numpy as np
import statistics

# drop rate 0.5, elu dense layer
class DenseLayer(Model):
    def __init__(self, n_units, drop_rate=.5, activation=tf.nn.elu):
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

def ConvLayer(inputs, drop_rate=0.3, filters=8, kernel_size=(1, 100), strides=(1, 10), padding='SAME'):
    conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                         kernel_initializer='he_normal',
                         kernel_regularizer=keras.regularizers.l2(0.01))(inputs)

    batchnorm = layers.BatchNormalization()(conv)
    activation = tf.nn.relu6(batchnorm)
    dropout = layers.Dropout(rate=drop_rate)(activation)

    return dropout

# CNN-LSTM model, 2 convolution layers connected to 3 LSTM layers
def CLNN(time_step, n_input, lstm_units):

    inputs = keras.Input(shape=(time_step, n_input, 1))
    conv1 = ConvLayer(inputs)
    # conv1 = ConvLayer()(inputs)

    #conv2 = ConvLayer(conv1)
    CNN_output = tf.reshape(conv1, (-1, conv1.shape[1], conv1.shape[2] * conv1.shape[3]))

    # for using GPUs
    #lstm1 = LSTMLayer(lstm_uits)(CNN_output)
    #lstm2 = LSTMLayer(lstm_uits)(lstm1)
    #lstm3 = LSTMLayer(lstm_uits)(lstm2)

    # without GPUs
    logits = LSTM(time_step, CNN_output.shape[2], lstm_units)(CNN_output)

    return Model(inputs=inputs, outputs=logits)

# input[0]과 input[1]의 attention score
class AttentionLayer(Model):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        layer1 = self.W1(inputs[0])
        layer2 = self.W2(inputs[1])
        score = tf.matmul(layer1, tf.transpose(layer2, [0, 2, 1]))
        dim = score.shape[2]
        score = score / np.sqrt(int(dim))
        distribution = tf.nn.softmax(score)
        att = tf.matmul(distribution, inputs[2])
        return att

# 3개의 LSTM layers로 구성된 model dropout 없음
def LSTM(n_timestep, n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5):

    inputs = keras.Input(shape=(n_timestep, n_inputs))
    dropout = layers.Dropout(rate = drop_rate)(inputs)
    lstm1 = layers.LSTM(n_units,
                        return_sequences=True,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(dropout)
    batchnorm1 = layers.BatchNormalization()(lstm1)

    lstm2 = layers.LSTM(400,
                        return_sequences=True,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm1)
    batchnorm2 = layers.BatchNormalization()(lstm2)

    lstm3 = layers.LSTM(200,
                        return_sequences=True,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm2)

    stacked_rnn_outputs = tf.reshape(lstm3, [-1, 200])
    stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
    logits = tf.reshape(stacked_outputs, [-1, n_timestep, 1])

    return Model(inputs=inputs, outputs=logits)

# 1개의 many-to-one LSTM layer로 구성된 model dropout 없음
def LSTMno_seq(n_timestep, n_inputs, n_units, regularizers_alpha=0.01):

    inputs = keras.Input(shape=(n_timestep, n_inputs))
    lstm = layers.LSTM(n_units,
                        return_sequences=False,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(inputs)

    stacked_rnn_outputs = tf.reshape(lstm, [-1, n_units])
    stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
    logits = tf.reshape(stacked_outputs, [-1, 1])

    return Model(inputs=inputs, outputs=logits)

# residual LSTM, input + 1'st LSTM output, 3'rd LSTM input + 3'rd LSTM output
def ResLSTM1(n_timestep, n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5, gpu=0):

    inputs = keras.Input(shape=(n_timestep, n_inputs))
    dropout = layers.Dropout(rate=drop_rate)(inputs)
    lstm1 = layers.CuDNNLSTM(n_inputs,
                             return_sequences=True,
                             kernel_initializer='he_normal',
                             kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(dropout)
    add1 = layers.add([lstm1, inputs])
    batchnorm1 = layers.BatchNormalization()(add1)

    lstm2 = layers.CuDNNLSTM(n_units,
                             return_sequences=True,
                             kernel_initializer='he_normal',
                             kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm1)
    batchnorm2 = layers.BatchNormalization()(lstm2)

    lstm3 = layers.CuDNNLSTM(n_units,
                             return_sequences=True,
                             kernel_initializer='he_normal',
                             kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm2)
    add2 = layers.add([lstm3, batchnorm2])
    batchnorm3 = layers.BatchNormalization()(add2)

    stacked_rnn_outputs = tf.reshape(add2, [-1, n_units])
    stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
    logits = tf.reshape(stacked_outputs, [-1, n_timestep, 1])

    return Model(inputs=inputs, outputs=logits)

# residual LSTM, 2'nd LSTM input + 2'nf LSTM output, input + 3'rd LSTM output
def ResLSTM2(n_timestep, n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5, gpu=0):

    inputs = keras.Input(shape=(n_timestep, n_inputs))
    dropout = layers.Dropout(rate=drop_rate)(inputs)
    lstm1 = layers.LSTM(n_units,
                        return_sequences=True,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(dropout)
    batchnorm1 = layers.BatchNormalization()(lstm1)

    lstm2 = layers.LSTM(n_units,
                        return_sequences=True,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm1)

    add1 = layers.add([lstm2, batchnorm1])

    batchnorm2 = layers.BatchNormalization()(add1)

    lstm3 = layers.LSTM(n_inputs,
                        return_sequences=True,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm2)

    add2 = layers.add([lstm3, inputs])

    batchnorm3 = layers.BatchNormalization()(add2)

    stacked_rnn_outputs = tf.reshape(batchnorm3, [-1, batchnorm3.shape[2]])
    stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
    logits = tf.reshape(stacked_outputs, [-1, n_timestep, 1])

    return Model(inputs=inputs, outputs=logits)

# ResLSTM2에 noise 추가, noise = random_normal(constnat) * random_normal(var) + random_normal(var)
def ResLSTM2noise(n_timestep, n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5, gpu=1):
    #with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(n_timestep, n_inputs))
        dropout = layers.Dropout(rate=drop_rate)(inputs)
        lstm1 = layers.LSTM(n_units,
                            return_sequences=True,

                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(dropout)
        batchnorm1 = layers.BatchNormalization()(lstm1)

        lstm2 = layers.LSTM(n_units,
                            return_sequences=True,

                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm1)

        add1 = layers.add([lstm2, batchnorm1])

        batchnorm2 = layers.BatchNormalization()(add1)

        lstm3 = layers.LSTM(n_inputs,
                            return_sequences=True,

                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm2)

        add2 = layers.add([lstm3, inputs])

        batchnorm3 = layers.BatchNormalization()(add2)

        stacked_rnn_outputs = tf.reshape(batchnorm3, [-1, batchnorm3.shape[2]])
        stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
        logits = tf.reshape(stacked_outputs, [-1, n_timestep, 1])

        # print('logits shape :', logits.shape[0])

        # sigma = tf.Variable(tf.random_normal([1,n_timestep,1]))
        # mu = tf.Variable(tf.random_normal([1,n_timestep,1]))
        sigma_init = tf.random_normal_initializer()
        mu_init = tf.random_normal_initializer()

        sigma = abs(tf.Variable(sigma_init(shape=(1, n_timestep, 1)), dtype='float32', name='sigma'))
        # sigma = keras.backend.get_value(sigma_init(shape = (1,120,1)))

        mu = tf.Variable(mu_init(shape=(1, n_timestep, 1)), dtype='float32', name='mu')
        print('sigma size :', sigma.shape)
        # noise = keras.layers.Dense(units = 1)(tf.random_normal(shape=tf.shape(logits), mean=0.0, stddev=1.0))
        print('logits shape:', logits.shape)
        print('tf.random.normal(shape=tf.shape(logits) :', tf.random.normal(shape=tf.shape(logits)).shape)
        random_noise = tf.random.normal(shape=tf.shape(logits))
        noise = tf.math.add(tf.math.multiply(random_noise, sigma), mu)
        print('noise shape :', noise.shape)
        logits_with_noise = layers.add([logits, noise])

        return Model(inputs=inputs, outputs=logits_with_noise)

# 5개의 comatible CuDNNLSTM layers, 각 layer 2 ~ 4에서 [layer input + layer output] layer 5에서는 [input + layer output]
def ResLSTM3(n_timestep, n_inputs, n_units, regularizers_alpha=0.01, drop_rate=0.5):

    inputs = keras.Input(shape=(n_timestep, n_inputs))
    dropout = layers.Dropout(rate=drop_rate)(inputs)
    lstm1 = tf.compat.v1.keras.layers.CuDNNLSTM(n_units,
                             return_sequences=True,
                             kernel_initializer='he_normal',
                             kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(dropout)
    batchnorm1 = layers.BatchNormalization()(lstm1)

    lstm2 = tf.compat.v1.keras.layers.CuDNNLSTM(n_units,
                             return_sequences=True,
                             kernel_initializer='he_normal',
                             kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm1)

    add1 = layers.add([lstm2, batchnorm1])

    batchnorm2 = layers.BatchNormalization()(add1)

    lstm3 = tf.compat.v1.keras.layers.CuDNNLSTM(n_units,
                             return_sequences=True,
                             kernel_initializer='he_normal',
                             kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm2)
    add2 = layers.add([lstm3, batchnorm2])

    batchnorm3 = layers.BatchNormalization()(add2)

    lstm4 = tf.compat.v1.keras.layers.CuDNNLSTM(n_units,
                             return_sequences=True,
                             kernel_initializer='he_normal',
                             kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm3)
    add3 = layers.add([lstm4, batchnorm3])

    batchnorm4 = layers.BatchNormalization()(add3)

    lstm5 = tf.compat.v1.keras.layers.CuDNNLSTM(n_inputs,
                             return_sequences=True,
                             kernel_initializer='he_normal',
                             kernel_regularizer=keras.regularizers.l2(regularizers_alpha))(batchnorm4)

    add4 = layers.add([lstm5, inputs])

    batchnorm5 = layers.BatchNormalization()(add4)

    stacked_rnn_outputs = tf.reshape(batchnorm5, [-1, batchnorm5.shape[2]])
    stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
    logits = tf.reshape(stacked_outputs, [-1, n_timestep, 1])

    return Model(inputs=inputs, outputs=logits)

# 2개의 convolution --> attention --> 3개의 LSTM
def CALNN(time_step, n_input, lstm_uits):

    inputs = keras.Input(shape=(time_step, n_input, 1))
    conv1 = ConvLayer(inputs)

    conv2 = ConvLayer(conv1)
    CNN_output = tf.reshape(conv2, (-1, conv2.shape[1], conv2.shape[2] * conv2.shape[3]))

    # print("CNN_output2:",CNN_output)
    attn = AttentionLayer(CNN_output.shape[2])([CNN_output, CNN_output, CNN_output])

    # print('attn1 :',attn1.shape)
    lstm1 = LSTMLayer(lstm_uits)(attn)
    lstm2 = LSTMLayer(lstm_uits)(lstm1)
    lstm3 = LSTMLayer(lstm_uits)(lstm2)

    # lstm1 = LSTMLayer(CNN_output,lstm_uits)
    # lstm2 = LSTMLayer(lstm1,lstm_uits)
    # lstm3 = LSTMLayer(lstm2,lstm_uits)

    stacked_rnn_outputs = tf.reshape(lstm3, [-1, lstm_uits])
    stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
    logits = tf.reshape(stacked_outputs, [-1, time_step, 1])

    return Model(inputs=inputs, outputs=logits)

def multi_input_CLNN(time_step, n_input1, n_input2, n_input3, lstm_uits):
    with tf.device('/gpu:0'):
        inputs1 = keras.Input(shape=(time_step, n_input1, 1))
        conv1_1 = ConvLayer()(inputs1)
        conv1_2 = ConvLayer()(conv1_1)
        conv1_output = tf.reshape(conv1_2, (-1, conv1_2.shape[1], conv1_2.shape[2] * conv1_2.shape[3]))

    with tf.device('/gpu:1'):
        inputs2 = keras.Input(shape=(time_step, n_input2, 1))
        conv2_1 = ConvLayer()(inputs2)
        conv2_2 = ConvLayer()(conv2_1)
        conv2_output = tf.reshape(conv2_2, (-1, conv2_2.shape[1], conv2_2.shape[2] * conv2_2.shape[3]))

    with tf.device('/gpu:2'):
        inputs3 = keras.Input(shape=(time_step, n_input3, 1))
        conv3_1 = ConvLayer()(inputs3)
        conv3_2 = ConvLayer()(conv3_1)
        conv3_output = tf.reshape(conv3_2, (-1, conv3_2.shape[1], conv3_2.shape[2] * conv3_2.shape[3]))

    with tf.device('/gpu:3'):
        cnn_output = tf.keras.layers.concatenate(inputs=[conv1_output, conv2_output, conv3_output], axis=-1)

        lstm1 = LSTMLayer(lstm_uits)(cnn_output)
        lstm2 = LSTMLayer(lstm_uits)(lstm1)
        lstm3 = LSTMLayer(lstm_uits)(lstm2)

        stacked_rnn_outputs = tf.reshape(lstm3, [-1, lstm_uits])
        stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
        logits = tf.reshape(stacked_outputs, [-1, time_step, 1])
    return Model(inputs=[inputs1, inputs2, inputs3], outputs=logits)


class PoolingAttentionLayer(Model):
    def __init__(self, size_group):
        super(PoolingAttentionLayer, self).__init__()
        self.size_group = size_group

    def call(self, inputs, training=False):
        self.groups = list()
        self.attn_li = list()
        value = inputs[0]  # conv_layer_output
        target = inputs[1]  # target_label
        target = tf.transpose(target, [0, 3, 1, 2])  # (batch, filter, time, feature =1)
        n_group = int(int(value.shape[2]) / self.size_group)  # number of group

        remainder = int(value.shape[2]) % self.size_group
        for i in range(0, n_group):  # split group
            self.groups.append(tf.slice(value, [0, 0, i * self.size_group, 0], [-1, -1, self.size_group, -1]))

        if remainder != 0:
            self.groups.append(tf.slice(value, [0, 0, n_group + 1 * self.size_group, 0], [-1, -1, remainder, -1]))

        for group in self.groups:  # compute attention by group
            group_trans = tf.transpose(group, [0, 3, 2, 1])
            # print('group shape : ', group_trans.shape)
            score = tf.matmul(group_trans, target)
            # print('score shape :', score.shape)
            attn = tf.nn.softmax(score)
            ##print(tf.transpose(group,[0,1,3,2]))
            # p#rint(tf.transpose(attn,[0,3,1,2]))
            self.attn_li.append(tf.matmul(tf.transpose(group, [0, 3, 1, 2]), attn))

            # self.attn_li.append(tf.transpose(group,[0,1,3,2])*tf.transpose(attn,[0,3,1,2]))
            # print('attn_li shape', len(self.attn_li))
        result = tf.concat(self.attn_li, axis=3)  # concat group

        result = tf.transpose(result, [0, 2, 3, 1])  # (batch, time,fature, filter)
        return result


def PoolingAttention(time_step, n_input, lstm_uits, gpu):
    #with tf.device('/gpu:' + str(gpu)):
        inputs = keras.Input(shape=(time_step, n_input, 1))
        target = keras.Input(shape=(time_step, 1, 1))

        # target = np.expand_dims(target, axis = -1)
        conv1 = ConvLayer(inputs, drop_rate=0.8, filters=8, kernel_size=(1, 100), strides=(1, 1), padding='SAME')

        attn1 = PoolingAttentionLayer(10)([conv1, target])
        print('conv1 shape :', conv1.shape, '  target shape : ', target.shape)
        print('attn1 shape : ', attn1.shape)
        conv2 = ConvLayer(attn1, drop_rate=0.8, filters=8, kernel_size=(1, 100), strides=(1, 1), padding='SAME')
        print('conv2 shape :', conv2.shape, '  target shape :', target.shape)
        attn2 = PoolingAttentionLayer(10)([conv2, target])
        CNN_output = tf.reshape(attn2, (-1, attn2.shape[1], attn2.shape[2] * attn2.shape[3]))

        lstm1 = layers.LSTM(lstm_uits,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(0.01))(CNN_output)
        lstm2 = layers.LSTM(lstm_uits,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(0.01))(lstm1)
        lstm3 = layers.LSTM(lstm_uits,
                            return_sequences=True,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(0.01))(lstm2)

        stacked_rnn_outputs = tf.reshape(lstm3, [-1, lstm_uits])
        stacked_outputs = keras.layers.Dense(units=1)(stacked_rnn_outputs)
        logits = tf.reshape(stacked_outputs, [-1, time_step, 1])

        return Model(inputs=[inputs, target], outputs=logits)

# 2개의 LSTM layer로 256feature encoding, 2개의 LSTM layer로 decoding
def AutoEncoderLSTM(n_timestep, n_inputs, regularizers_alpha=0.01, drop_rate=0.5, gpu=0):
    #with tf.device('/gpu:' + str(gpu)):
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


def loss_mean_square(model, base_data, input_data, output_data):
    logits = model(input_data, training=True)
    loss = tf.reduce_mean(tf.square(logits - output_data))
    return loss


def loss_cross_entropy(model, base_data, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels))
    return loss


def loss_add_sharp_ratio(model, base_data, input_data, output_data):
    prediction_data = model(input_data, training=True)
    profit_rates = np.sign(prediction_data) * output_data / base_data # 수익률 list
    profit_rate_mean = tf.reduce_mean(profit_rates) # 수익률 평균
    std = statistics.stdev(profit_rates) # 수익률의 표준 편차
    sharp_ratio = profit_rate_mean / std
    loss = tf.reduce_mean(tf.square(prediction_data - output_data)) + sharp_ratio
    return loss


def loss_add_variance(model, base_data, input_data, output_data):
    prediction_data = model(input_data, training=True)
    profit_rates = np.sign(prediction_data) * output_data / base_data # 수익률 list
    var = statistics.variance(profit_rates) # 수익률의 분산
    loss = tf.reduce_mean(tf.square(prediction_data - output_data)) + var
    return loss

def loss_prediction_interval(model, input_data, prediction, real):
    std = tf.reshape(model(input_data, training=True), [-1]) * 1000
    prob = (1 / tf.sqrt(2 * np.pi * std * std)) * tf.exp(-tf.square(real-prediction) / tf.square(std))
    loss = tf.reduce_mean(-tf.log(prob))
    return loss

def gradient(model, type, base_data, input_data, output_data):
    with tf.GradientTape() as tape:
        if type == "cross_entropy" : loss = loss_mean_square(model, None, input_data, output_data)
        elif type == "add_sharp_ratio" : loss = loss_mean_square(model, None, input_data, output_data)
        elif type == "add_variance" : loss = loss_mean_square(model, base_data, input_data, output_data)
        elif type == "prediction_interval" : loss = loss_prediction_interval(model, base_data, input_data, output_data)
        else : loss = loss_mean_square(model, None, input_data, output_data)
    return tape.gradient(loss, model.trainable_variables)

def compute_apply_gradients(model, loss, optimizer):
  gradients = gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def evaluate(model, input_data, output_data):
    logits = model.predict(input_data)
    MSE = tf.reduce_mean(tf.square(logits[:, -1, -1] - output_data[:, -1, -1]))
    return MSE, logits

