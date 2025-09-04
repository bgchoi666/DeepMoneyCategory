# version_name
ver = "20norm_1days_high"

#data file directorty
read_dir = ""

#data file name
file_name = "../fnguide data/kospi200f_20norm_0304_high.csv"
market = "kospi200f"

#temporary file name
tmp_file_name = "tmp.csv"

#model directory
model_dir = "model_dir/" + ver

#result file directory
result_dir = "results_" + ver

#log file name
log_dir = "log/" + ver + ".log"

# the number of threads
num_threads = 4

class Config(object):

    ensemble = False
    gradual = True

    init_scale = 0.05
    learning_rate = 0.001

    predict_term = 1

    step_interval = 1
    num_steps = 20

    num_layers = 2
    hidden_size = 500
    batch_size = 20

    input_size = 943
    output_size = 1

    rnn_mode = "basic"
    norm_days = 0
    conversion = 'diff'

    iter_steps_max = 500
    iter_steps = 100
    gradual_steps = 500

    alpha = 0
    beta = 0

    model_reset = True
    shuffle = True

    train_start1 = "2000-01-03"
    train_end1 = "2015-12-31"

    train_start2 = "2000-01-03"
    train_end2 = "2017-12-31"

    test_start = "2018-01-01"
    test_end = "2019-01-01"

    grad_train_terms = ["2019-01-01", "2018-01-01", "2018-02-01",
                        "2018-03-01", "2018-04-01", "2018-05-01", "2018-06-01",
                        "2018-07-01", "2018-08-01", "2018-09-01", "2018-10-01", "2018-11-01", "2018-12-01", "2019-01-01"]#, "2019-02-01",
                        #"2019-03-04"]
                        #"2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01", "2018-05-01", "2018-06-01",
                        #"2018-07-01", "2018-08-01", "2018-09-01", "2018-10-01"]#, "2018-11-01"]

    grad_train_test_terms = [
                        # train start  ["2000-01-03", "2011-01-01", "2012-01-01",
                        ["2000-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01", "2018-05-01", "2018-06-01", "2018-07=01"],
                        # train end    ["2010-12-31", "2011-12-31", "2012-12-31",
                        ["2017-12-31", "2018-01-31", "2018-02-28", "2018-03-31", "2018-04-30", "2018-05-31", "2018-06-30", "2018-07-31", "2018-08-31", "2018-09-30", "2018-10-31", "2018-11-30"],
                        # test start   ["2011-01-03", "2012-01-03", "2013-01-03",
                        ["2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01", "2018-05-01", "2018-06-01", "2018-07-01", "2018-08-01", "2018-09-01", "2018-10-01", "2018-11-01", "2018-12=01"],
                        # test end     ["2011-12-31", "2012-12-31", "2013-12-31",
                        ["2018-01-31", "2018-02-28", "2018-03-31", "2018-04-30", "2018-05-31", "2018-06-30", "2018-07-31", "2018-08-31", "2018-09-30", "2018-10-31", "2018-11-30", "2018-12-31"],
                        ]
    def __init__(self):
        if self.predict_term == 1:
           self.num_steps_list = [10, 20, 10, 20],
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
