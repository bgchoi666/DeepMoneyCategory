# version_name
ver = "v5_0_1_809_미래"

#data file directorty
read_dir = ""

#data file name
file_name = "../DeepMoneyData/current/kospi200f_809_0515.csv"
market = "kospi200f_809"

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

# model_path
#model_dirs = ["kospi200f_1_1_20_500_20_20190825172755", "kospi200f_1_1_20_500_20_20190825124439", "kospi200f_1_1_20_500_20_20190825104610", "kospi200f_1_1_20_500_20_20190824071612"]

class Config(object):

    ensemble = False
    gradual = True

    # reinforcing mode
    rmode = False
    base = "open"

    init_scale = 0.05
    learning_rate = 0.001

    predict_term = 5

    step_interval = 1
    num_steps = 20

    num_layers = 2
    hidden_size1 = 500
    hidden_size2 = 5
    batch_size = 20

    input_size = 809
    output_size = 1

    rnn_mode = "basic"
    norm_days = 0
    conversion = 'diff'

    iter_steps_max = 1000
    iter_steps = 100
    gradual_steps = 500
    gradual_term = 1

    alpha = 0.5 # the weight of RRL loss
    beta = 0.1 # the weight of kelly loss in non-gradual mode

    model_reset = True
    shuffle = True

    train_start1 = "2000-02-10"
    train_end1 = "2020-05-15"

    train_start2 = "2000-02-10"
    train_end2 = "2019-01-04"

    test_start = "2019-01-02"
    test_end = "2019-01-03"

    train_start1_index = 0
    train_end1_index = 0

    train_start2_index = 0
    train_end2_index = 0

    test_start_index = 0
    test_end_index = 0

    def __init__(self):
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
