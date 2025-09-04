# version_name
ver = "20norm_1days_v4_0"

#data file directorty
read_dir = ""

#data file name
file_name = "../DeepMoneyData/kospi200f_20norm_current.csv"
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
    gradual = False
    RRL = True
    kelly = False # newly added property in v4.0

    init_scale = 0.05
    learning_rate = 0.001

    predict_term = 1

    step_interval = 1
    num_steps = 20

    num_layers = 2
    hidden_size1 = 500
    hidden_size2 = 5
    batch_size = 20

    input_size = 943
    output_size = 1

    rnn_mode = "basic"
    norm_days = 0
    conversion = 'diff'

    iter_steps_max = 500
    iter_steps = 100
    gradual_steps = 100
    gradual_term = 65

    alpha = 0
    beta = 0

    model_reset = True
    shuffle = True

    train_start1 = "2000-01-03"
    train_end1 = "2008-12-31"

    train_start2 = "2001-01-03"
    train_end2 = "2001-01-03"

    test_start = "2009-01-01"
    test_end = "2013-12-31"

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
