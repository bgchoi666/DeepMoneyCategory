# DeepMoney version 2.x.x description

The data consisted of 72 to 626 items received daily from FNGUIDE from January 25, 1997 to January 25, 2017, and was trained and tested.
The model structure used two LSTM layers, and the final output was trained using the tanh function to produce values ​​close to -1 for declines and close to 1 for increases.
The model was completed by reconfiguring an example downloaded from the Maven repository using the IntelliJ development tool for futures index prediction.

Let me explain one of the test results.

The model contained 326 items, including index and other market information, economic indicators, overseas markets (excluding Europe and the US), and exchange rate information.
From January 4, 2016, to August 26, 2016, the average monthly return was 3.71 points, with a win rate of 58.75%. Of course, tests on data including overseas indices yielded a success rate of over 70%. However, this is likely due to the fact that Western markets, such as the US, close a day later than ours, so retroactive records are stored, resulting in past data including information from the day the prediction is being made.
* Item list
COMPANY Desc Stock-Exchange-Purchase Quantity (-Registered Foreigners) (Thousand Shares) Stock-Exchange-Purchase Quantity (Institutional) (Thousand Shares) Stock-Exchange-Purchase Quantity (Financial Investment) (Thousand Shares) Stock-Exchange-Purchase Quantity (-Insurance) (Thousand Shares) Stock-Exchange-Purchase Quantity (-Investment Trust) (Thousand Shares)
… … ... ..
Stock Index Futures - Futures (SP Sum) - Sell Quantity (Financial Investment) (Contract) Stock Index Futures - Futures (SP Sum) - Sell Amount (Financial Investment) (Million KRW) Stock Index Futures - Futures (SP Sum) - Net Buy Quantity (Financial Investment) (Contract) Stock Index Futures - Futures (SP Sum) - Net Buy Amount (Financial Investment) (Million KRW)
… … ... … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … ..
Interest Rates and Commodity Futures - EUR Futures - Net Purchase Amount (-Securities/Futures) (Millions of KRW) Interest Rates and Commodity Futures - 3-Year Government Bonds (Excluding SP) - Purchase Amount (-Securities/Futures) (Contracts) Interest Rates and Commodity Futures - 3-Year Government Bonds (Excluding SP) - Sell Amount (-Securities/Futures) (Contracts) Interest Rates and Commodity Futures - 3-Year Government Bonds (Excluding SP) - Net Purchase Amount (-Securities/Futures) (Millions of KRW)
… … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … . Exchange-program trading net purchase amount (million won) Exchange-arbitrage transaction sales quantity (thousand shares) Exchange-arbitrage transaction sales amount (million won) Exchange-arbitrage transaction purchase quantity (thousand shares) Leading composite index (2010=100) Leading composite index month-on-month comparison (%) Inventory circulation index (%) Consumer expectations index (2010=100) Machinery domestic export index (excluding ships) (2010=100) Construction orders (real) (billion won) Export-import price ratio (%) International raw material price index (inverse series) (2010=100) Job openings-seekers ratio (%) KOSPI index (1980.1.4=100)(1980.1.4=100) Short-term and long-term interest rate differential (%) Coincident composite index (2010=100) Mining and manufacturing production index (2010=100) Coincident Composite Index Previous Month Comparison (%) Coincident Index Cyclical Component (%) Service Industry Production Index (excluding wholesale and retail trade) (2010=100) Construction Completed Amount (Real) (Billion KRW) Retail Sales Index (2010=100) Domestic Exports Index (2010=100) Import Amount (Real) (Million USD) Number of Non-Agricultural, Forestry and Fisheries Employees (Thousands) Lagging Composite Index (2010=100) Producer Product Inventory Index (2010=100) Urban Household Consumption Expenditure (Real) (Thousands KRW) Lagging Composite Index Previous Month Comparison (%) Import Amount (Real) (Million USD) Number of Regular Employees (Thousands) Corporate Bond Distribution Yield (%) Industrial Production Index (Won Index) (2010=100) Industrial Production Index (Seasonally Adjusted) (2010=100) Producer Products Shipment Index (KRW Index) (2010=100) Producer Product Shipment Index (Seasonally Adjusted) (2010=100) Producer Product Inventory Index (KRW Index) (2010=100) Producer Product Inventory Index (Seasonally Adjusted) (2010=100) Domestic Export Index (KRW Index) (2010=100) Domestic Export Index (Seasonally Adjusted) (2010=100) Export Export Index (KRW Index) (2010=100) Export Export Index (Seasonally Adjusted) (2010=100) Production Capacity Index (2010=100) Operation Rate Index (KRW Index) (2010=100)
… … ... Currency Issue (Billions of Won) Market Interest Rate: 3-Year Treasury Bonds (3-Year National Debt Management Fund Bonds) (%) Total (Thousand Dollars) Total Imports by Country (Thousand Dollars) Exports by Country Japan (Thousand Dollars) Exports by Country China (Thousand Dollars) Exports by Country United States (Thousand Dollars)
… … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … … ..
Market Average_US Dollar (KRW) Market Average_Japan (100 Yen) ((100) Currency KRW) Hong Kong Foreign Exchange Fund Bond (10 Years) (%) Korea WON (Foreign Exchange Brokerage Closing Price) (USD KRW) US SP 500 Index (Closing Price) (Pt) US Nasdaq 100 (Closing Price) (Pt) US Dow Jones 30 Industrials (Closing Price) (Pt) Japan Nikkei 225 (Closing Price) (Pt) UK FTSE 100 (Closing Price) (Pt) Major Commodity Index_CRB Index (Futures) (Pt) Major Commodity Index_CRB Index (Spot) (Pt) Soybean Meal (￠/bu) Open Index (Points) High Index (Points) Low Index (Points) Closing Index (Points) Return (%) Return (1 Week) (%) Return (1 Month) (%) Return (3 Months) (%) Return (6 months) (%) Return (12 months) (%) Closing index (5-day high) (points) Closing index (5-day low) (points) Closing index (5-day average) (points) Closing index (20-day high) (points) Closing index (20-day low) (points) Closing index (20-day average) (points) Closing index (60-day high) (points) Closing index (60-day low) (points) Closing index (60-day average) (points) Closing index (120-day high) (points) Closing index (120-day low) (points) Closing index (120-day average) (points) Closing index (52-week high) (points) Closing index (52-week low) (points) Closing index (52-week average) (points) Trading volume (shares) Trading amount (won) Trading amount (5-day average) (won) Trading amount (20-day average) (won) Trading amount (60-day average) (won) Trading amount (120-day average) (KRW) Trading volume (52-week average) (KRW) Foreign ownership percentage (%) Foreign market capitalization (KRW million) "Market capitalization (52-week low, average) (KRW million)" "Market capitalization (52-week average, average) (KRW million)" "Market capitalization (YTD average, average) (KRW million)" Open price (points) High (points) Low (points) Closing price (points) Theoretical price (points) Reference price (points) Underlying asset price (points) Trading volume (contracts) Trading value (KRW thousand) Remaining days (days) Open interest (contracts)

Personal opinion:
- Developers familiar with Java can easily create a prediction system. For example, refer to the RNN-related example among the various deep learning examples in DL4J. However, compared to Python, the data processing library ND4J is somewhat less user-friendly than pandas.

Tensorflow-LSTM-FNGUIDE: This tool receives data from FNGUIDE and analyzes it using deep learning (Tensorflow with LSTM) for up to 943 items. It primarily uses TensorFlow low versions 1.1 to 1.8. While several versions exist under this title, the differences are not significant.

- Created in 2018 during the development of Shinhan's Treasure Island project, using a similar algorithm and FNGUIDE data (up to 943 items).

range(1, 39) # Open, Low, Close, Return, Moving Average, . . range(79, 147) # Economic Index
range(195, 260) # Money Supply, Interest Rates
range(260, 314) # Exports and Imports, Current Account Balance, Investment, Foreign Exchange Reserves, Assets, External Debt
range(314, 334) # Exchange Rates
range(406, 454) # Interest Rates, Government Bond Yields
range(456, 561) # Foreign Stock, Commodity, and Raw Material Indices
range(561, 565) # Economic Growth Rates of Korea, the US, Japan, and the UK
range(565, 934) # Financial Asset Status, Trading Volume
range(934, 938) # Open, Low, and Closing Prices
range(938, 945) # Theoretical Value, Reference Price, Underlying Asset Price, Trading Volume (Amount), Remaining Days, Open Interest

- Selectively Use LSTM and GRU

- 5, 20, and 65 Days Regression on post-index predictions.
* 5-day predictions typically recorded an accuracy rate of around 53%, 20-day predictions in the low 60s, and 65-day predictions in the high 60s to low 70s. The highest return was around 150% over the one-year period from July 2017 to June 2018.

- MDD is extremely high. I haven't calculated it exactly, but it's probably around 70-80%. In other words, when losses occur, they are terribly forgotten.

- 4 versions
* diff: Target data is the index on the forecast date - the index on the reference date. Options include rate and norm.
* 4 inputs: 1-day, 5-day, 20-day, and 65-day prediction targets in one file.
* bootstrap
* gradual_train: Divide the training data into multiple intervals and train them sequentially.

# missing point:
If the entry point for the 1-day and 2-day predictions is the closing price, the test results will be better than the actual results. This is because the timing of overseas indices doesn't align with other factors. While overseas indices are determined around 6:00 AM the following day, data is assumed to have been entered at the previous day's domestic market close, reflected in the previous day's closing price, and processed during testing. In other words, it becomes future data. This essentially predicts the future, which inevitably leads to better predictions. Later, the test was changed to forecast based on the closing price, but entry is made at the opening price of the following day.

- Missing point: If the entry point is the closing price in the 1-2 day forecast, the test result will be better than the actual result. This is because the timing of the overseas index does not match that of other items. The overseas index is determined around 6:00 AM the following day, but the data is considered to have been entered at the closing price of the previous day's domestic market, so it is reflected in the previous day's closing price and processed during the test. In other words, it becomes future data. This is like predicting the future in advance, so the prediction result is bound to be good. Later, the test is changed to forecasting at the closing price but entering at the opening price of the following day.

Deepmoney version 3 includes two major improvements:

- During testing, profit and loss calculations are made based on the closing price of the day, but entry is made at the opening price of the next day.

- The sharp ratio value is subtracted from the loss function. After one training step, profit and loss can be calculated based on the model's predictions for the test period. Here, the sharp ratio value for downside returns is subtracted from the mean square error of the prediction.

- Some examples of experimental results for predicting a two-day entry at the opening price of the next day:
* 839% profit based on the closing price from 2018-01-01 to 12-31, 65% profit based on the opening price of the next day
* 866% profit based on the closing price from 2018-01-01 to 12-31, 66% profit based on the opening price of the next day

- The significant difference in returns between closing and opening price entries stems from the difference in whether or not the future outcome can be predicted. If only we could predict the future, what would be the point of worrying?

- Starting with version 4, the Kelly formula is applied to adjust the loss value to maximize (average profit - average loss).

- A representative experimental result was a strategy that predicted the next day's closing price and entered at the next day's opening price over a 12-year period from 2007 to 2018. It achieved an average annual return of 42%. However, two losing years occurred, and the minimum drawdown (MDD) was also significant.

Version 5 adds RRL (the sharp ratio of downside return) and Kelly (average profit - average loss) to the loss value to maximize the ratio. Experiments were conducted from 2007 to 2018 using a strategy that predicted the next day's closing price from the current day's closing price, entered at the next day's opening price, and liquidated at the next day's closing price.

- Average evaluation of multiple experimental results: For a 40% investment ratio, the average annual compound return was 57%.

In Version 6, predictions were made based on the next day's opening price instead of the closing price.
- No good experimental results.

 Versions up to and including Version 6 were developed using TensorFlow version 1.x. They used an estimator function that evaluated the formula using Session. The structure was quite complex and difficult to understand. However, the model performed relatively well. In particular, adding RRL (sharp ratio) and Kelly (average profit/loss difference) to the loss function showed a significant performance improvement compared to the previous version.

 When running Version 5 on TensorFlow 1.13 or 2.0, the results showed a hit rate of 48% to 55% based on the next day's opening price when testing one-day predictions for approximately one year from 2019 to February 2020. Based on closing prices, the prediction accuracy rate was 52-55%. The data set consisted of 1,050 items, including international indices.

# Deepmoney v5.0.1-tf1.13
Execution Environment: TensorFlow 1.13
Overview: A model function is built using compatible cudnn LSTM. Data feeding, training, and testing of the built model function are automated using a customized model estimator. The data is a 3D dataset (.csv) of daily stock price, economic, and market information time series data downloaded from fnguide's dataguide, formatted as (batches, steps, inputs) and (batches, steps, target). This dataset is then input to the input function as data for train, test, and predict. The entire dataset is divided into a train dataset and a test dataset. Data from the prediction attempt date to the prediction target date is not used as the train target. For prediction input, time series data from a certain period (steps) after the prediction attempt date is sequentially input to the LSTM steps. The data is updated to minimize MSE loss so that the output at each time point corresponds to the corresponding target. After training, the test results are saved in a CSV file, including the prediction attempt date, target date, index of the prediction attempt date, actual index, predicted index, and profit according to the prediction result.

# Deepmoney7 - tansfer_LSTM2
Execution Environment: TensorFlow 2.1
Overview: Extends the model from version 5.0.1-tf2. It has 3 layers and 800 units per layer. Training is performed using Keras model compile and fit. The entire dataset is divided into batches, deeded, and each batch is allocated to a different number of GPUs using a mirrored strategy. Training is performed in parallel. The training loss value is the same as v5.0.1-tf2.

# Deepmoney7 - transfer_LSTM2-copy2
Execution environment: TensorFlow 2.1
Overview: Extends the transfer_LSTM2 model. It has two layers and 1,024 units per layer. It trains using Keras's compile and fit functions. The difference from the original model is that the test period runs from 2010 to May 15, 2020. For each test period, the training period (1,000 days prior to the start of the test) is randomly divided into two parts, with only one used. This process is repeated 10 times with different models, and the best-performing model is selected. This process continues for one year.

# Deepmoney7 - transfer_LSTM2-copy3
Execution environment: TensorFlow 2.1
Overview: Extends the transfer_LSTM2 model. It has two layers and 1,024 units per layer. Training is performed using Keras's compile and fit functions. The difference from the original model is that the test period runs from 2010 to May 15, 2020. For each test period, the training period (1,000 days prior to the start of the test) is randomly divided into two parts, with only one used. This process is repeated 10 times with different models, and the best-performing model is selected. This process is repeated for 10 days each.

# Deepmoney8 – ddaeryuble2
Execution environment: TensorFlow 2.1
Two models are trained alternately. Model 1 is the prediction model, and Model 2 is the calibration model, determining the appropriate investment ratio and direction (-1 to 1).

# Deepmoney8 – ddaeryuble2-1
Execution environment: TensorFlow 2.1
The difference from ddaeryuble2 is that Model1 is first trained using compile and fit, and then Model2 is trained using a random interval selection method, sequentially selecting a batch of data.

# Deepmoney8 – ddaeryuble3
Execution environment: TensorFlow 2.1
Training is performed alternately using two models. Model1 is the prediction model, and Model2 serves as the calibration model, determining the appropriate investment ratio and direction (-1 to 1).

# Deepmoney8 – ddaeryuble5
Execution environment: TensorFlow 2.1
Two models are trained alternately. Model 1 is a prediction model, and Model 2 is a model for predicting the expected return on the target date. The training loss for Model 1 is the MSE between the sign of the output and the actual return. The training loss for Model 1 is the MSE targeting the geometric mean of the cumulative predicted return on the target date, calculated from the sequence of actual and predicted returns at the current point in time. The prediction consists of two values: a simple prediction by Model 1 and a predicted value adjusted to maximize the output of Model 2.

# Deepmoney8 – ddaeryuble5-1
Execution environment: TensorFlow 2.1
The difference from ddaeryuble5 is that instead of training Model 1 and Model 2 alternately, Model 1 training is completed before Model 2 is trained. The inputs for Model 2 are the predicted and actual returns, and the target is the profit of the next batch (spread out by the prediction period).

# Deepmoney8 – ddaeryuble5-2
Execution environment: TensorFlow 2.1
The difference from Ddaeryuble5-1 is that Model 1 uses the compile and fit methods for training.

# Deepmoney8 – ddaeryuble6
Execution environment: TensorFlow 2.1
Model 1 is built as a prediction model, and Model 2 is built as an evaluation model. Model 2 adds a sequence of Model 1's predictions to the existing input. It updates the predicted return as its output, and sets the target as (the return after the current prediction period + 1) × (the ideal return for the next prediction period + 1) × α (0 to 1). Then, among several values ​​between the - absolute value and + absolute value of Model 1's prediction, it selects the predicted value that maximizes Model 2's output and updates Model 1. This interactive learning process between Model 1 and Model 2 is repeated.

# Deepmoney8 – ddaeryuble6-4
Execution environment: TensorFlow 2.1
Model 1 is an LSTM prediction model (sign), and Model 2 is a DNN evaluation model. Model 2 uses the closing prices at each step of Model 1 as variables. In other words, if Model 1 has 100 sequence steps, Model 2 also has 100 input variables. Model 1 is first trained with random train set 1, and then Models 1 and 2 are trained alternately with random train set 2.

# Deepmoney9 – loss_cut
Execution environment: TensorFlow 2.1
Model 1 is an LSTM prediction model that minimizes the MSE (Mean Severity Error) from the target rate of return (%) and maximizes the difference between the average profit and average loss as its loss. Model 2 outputs the optimal stop-loss value when investing based on Model 1's predictions, and its input is a combination of Model 1's input and its output.

# Deepmoney9 – loss_cut-4
Execution Environment: TensorFlow 2.1
The difference from loss-cut-3 lies in the change in the loss_fn_model2 function, which calculates the loss value for model2. While loss_cu-3 directly targets the stop-loss value and determines the loss value as the MSE with model2, loss_cut-4 predicts model2's output for each input in the batch and applies that output (stop-loss value) as the stop-loss value based on the predicted results for each input in the batch. This training method maximizes the average stop-loss profit for the resulting batch.
