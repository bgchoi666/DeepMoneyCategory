# version_name
ver = "totnorm_0304_65days"

#data file directorty
read_dir = ""

#data file name
file_name = "../fnguide data/kospi200f_totnorm_0304.csv"
market = "kospi200f"

#model directory
model_dir = "model_dir/" + ver

#result file directory
result_dir = "results_" + ver

#log file name
log_dir = "log/" + ver + ".log"

class Config(object):

    init_scale = 0.05
    learning_rate = 0.001
    num_layers = 2
    num_steps = 20
    hidden_size = 500
    batch_size = 30
    input_size = 943
    output_size = 1
    rnn_mode = "basic"
    conversion = 'diff'
    iter_steps_max = 1000
    iter_steps = 100
    step_interval = 10
    train_start = "2000-01-01"
    grad_train_terms = ["2000-01-01", #"2016-10-01", "2016-11-01", "2016-12-01", "2017-01-01", "2017-02-01", "2017-03-01", "2017-04-01", "2017-05-01",  "2017-06-01", "2017-07-01",

                        #"2017-08-01", "2017-09-01",
                        #"2017-10-01", "2017-11-01", "2017-12-01",
                        #"2018-01-01", "2018-02-01",
                        "2018-03-01", "2018-04-01", "2018-05-01", "2018-06-01",
                        "2018-07-01", "2018-08-01", "2018-09-01", "2018-10-01", "2018-11-01",
                        "2018-12-01", "2019-01-01", "2019-02-01", "2019-03-04"]
    test_start = "2018-03-01"
    test_end = "2019-03-04"
    predict_term = 65
    model_reset = True
    shuffle = True
    read_mode = "norm" # raw : read raw data
    alpha = 0.001
    beta = 0.001

    def __init__(self):
        if self.predict_term == 1:
           self.num_steps_list = [10, 20, 10, 20]
           self.step_interval_list = [1, 1, 2, 2]
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
