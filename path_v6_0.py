# version_name
ver = "v6_0_809"

#data file directorty
read_dir = ""

#data file name
file_name = "../DeepMoneyData/current/kospi200f_809.csv"
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
num_threads = 1

# model_path
# model 786
#model_dirs = ["kospi200f_1day_alpha0.5_beta0_20190924230208"]

#model 787
#model_dirs = ["kospi200f_1day_0.5_0_T_20191007073259"]

#model 809
model_dirs = ["kospi200f_1day_0.5_0_T_20191008065313"]

#model 810
#model_dirs = [""]

class Config(object):

    mode = 'p'
    rmode = False

    ensemble = False
    gradual = True

    # reinforce the model trained in gradual node

    init_scale = 0.05
    learning_rate = 0.001

    predict_term = 1

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

    iter_steps_max = 500
    iter_steps = 100
    gradual_steps = 300
    gradual_term = 365

    alpha = 0.5 # the weight of RRL loss
    beta = 0.1 # the weight of kelly loss in non-gradual mode

    model_reset = True
    shuffle = True

    train_start1 = "1996-07-01"
    train_end1 = "2018-12-31"

    train_start2 = "1997-01-01"
    train_end2 = "1998-12-31"

    test_start = "2019-10-01"
    test_end = "2019-11-30"

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
