# version_name
ver = "65days"
var_upd = "_totnorm_1mup_0101"

#data file directorty
read_dir = "C:/Users/Admin/Desktop/DeepMoney_v2.0/"

#data file namer
file_name = "../fnguide data/kospi200f_totnorm_0304.csv"
market = "kospi200f"

#model directory
model_dir = "C:/Users/Admin/Desktop/DeepMoney_v2.0/model_dir/" + ver + var_upd

#result file directory
result_dir = "C:/Users/Admin/Desktop/DeepMoney_v2.0/results_" + ver + var_upd

#log file name
log_file_name = "C:/Users/Admin/Desktop/DeepMoney_v2.0/log/deepmoney.log"

class Config(object):

    init_scale = 0.05
    learning_rate = 0.001
    num_layers = 2
    num_steps = 20
    hidden_size = 500
    batch_size = 20
    input_size = 943
    output_size = 1
    rnn_mode = "basic"
    conversion = 'diff'
    iter_steps = 1000
    step_interval = 10
    train_start =  "2000-01-01"
    grad_train_terms = ["2000-01-01", #"2017-10-01", "2017-11-01", "2017-12-01",
                        "2018-01-01", "2018-02-01",
                        "2018-03-01", "2018-04-01", #"2017-07-01",   "2017-08-01", "2017-09-01",
                        "2018-05-01", "2018-06-01", "2018-07-01", "2018-08-01", "2018-09-01",
                        "2018-10-01", "2018-11-01", "2018-12-01", "2019-01-01"]
                        #"2019-02-01", "2019-03-04"]
    test_start = "2018-03-01"
    test_end = "2019-03-04"
    predict_term = 65
    model_reset = True
    shuffle = True
    read_mode = "norm" # raw : read raw data
    alpha = 0.001
    beta = 0.001