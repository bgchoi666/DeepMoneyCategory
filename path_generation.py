# version_name
ver = "generation"
var_upd = "totnorm_0101_1031"

#data file directorty
read_dir = "C:/Users/Admin/Desktop/DeepMoney_v2.0/"

#data file namer
file_name = "kospi200f_totnorm_1031.csv"
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
    num_steps = 30
    hidden_size = 200
    batch_size = 20
    input_size = 943
    output_size = 1
    target_index = 938
    rnn_mode = "basic"
    conversion = 'norm'
    iter_steps = 300
    step_interval = 2
    train_start =  "2000-01-01"
    grad_train_terms = ["2000-01-01", "2018-01-01", #"2018-02-01", "2018-03-01", "2018-04-01", "2017-07-01", "2017-10-01","2017-11-01", "2017-12-01",  "2017-08-01", "2017-09-01",
                        #"2018-05-01",  "2018-06-01", "2018-07-01", "2018-08-01",
                        "2018-09-22"]#"2018-06-25"  , "018-09-01"
    test_start = "2017-04-03"
    test_end = "2017-05-01"
    predict_term = 5
    model_reset = True
    shuffle = True
    read_mode = "norm" # raw : read raw data
    alpha = 0.001
    beta = 0.001