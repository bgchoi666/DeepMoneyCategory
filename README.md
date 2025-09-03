# DeepMoney
the collection of kospi200 futures prediction projects using tensorflow, LSTM, CNN, DNN, reinforcement learning
Overview and History of Prediction and Trading Systems

A brief description of various algorithms developed for KOSPI 200 futures index prediction and trading using deep learning from 2016 to the present.

# DL4J-LSTM-FNGUIDE-n-Days-Ahead
Closing Price Prediction-Night Futures-Manual Trading: 5-10 minutes before 6:00 PM every day after the market closes, fnguide data is received and analyzed using a program trained on the DL4J deep learning platform. The system then predicts the next day's futures index closing price, enters the position at the 6:00 PM overnight futures opening, and liquidates the position at the closing price n days later. Manual Trading

# Tensorflow-LSTM-FNGUIDE-n-Days-Ahead
Closing Price Prediction-Next-Day Opening Entry-Manual Trading: FNGUIDE data is downloaded at 5-6:00 AM the following day after the market closes, when US and European index data (S&P 500, FTSE, DAX, etc.) arrive. Using a program trained on the Tensorflow platform, the system makes predictions and enters the position at the opening of the morning market that day. Liquidation at the closing price after n days. Manual trading.

There are several versions (versions 1 to 7), and the data download time, liquidation time, and number of data items may vary slightly.

# Tensorflow-LSTM-RRL-FNGUIDE-Predict closing price after n days
- Enter opening price after the next day - Manual trading: Same as above. The difference is that during training, the loss value is minus the shap ratio for downside returns.

# Tensorflow-LSTM-RRL-Kelly-FNGUIDE-Predict closing price after n days
- Enter opening price after the next day - Manual trading: Same as above. The difference is that during training, the loss value is minus the shap ratio for downside returns, and the average profit rate is minus the average loss rate. Notebook version available.

# Transfer_LSTM: 
Rebuilt as a notebook based on the source code written by Seong-Gil Cheon.

# Transfer learning:
Divide the training interval into multiple intervals and train sequentially from the first to the most recent interval. After training for one interval, the model is not reset and continues to train for the next interval.
Confusion matrix evaluation: True Positive, True Negative, False Positive, and False Negative are evaluated based on whether the target and predicted prices are aligned with the reference price.
Loss: sharp ratio, variance

# Transfer_LSTM2-DNN:
Shares submodules with Transfer_LSTM, but uses a DNN model instead of an LSTM.
The entire data set, excluding the test set, is repeatedly randomly split into a train set and an evaluation set, and the model with the best accuracy is saved.
Loss: negative sharp ratio, positive maximum drawdown

# Reinforcement learning (ddaeryuble series): 
Consists of two LSTM models. One model predicts the index after n days. The other model evaluates the return on the predicted values ​​or outputs the optimal loss-cut. Alternating training of the two models did not yield any improved results.

# High-Low Model: 
Determines whether the current candlestick is a high or low, and adopts a neutral position if it's neither high nor low. It uses daily and 60-minute data. The train model randomly samples the data set, splitting it into train and evaluation stages. The train stage is then repeated multiple times (more than 100 times) to select the model with the highest accuracy.
