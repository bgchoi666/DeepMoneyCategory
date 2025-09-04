# version_name
ver = "20170701"

#data file directorty
read_dir = "C:/Users/Admin/Desktop/DeepMoney/"

#file name
file_name = "s&p500-752-tot_norm.csv"

# market name
market = "s&p500"

#model directory
model_dir = "C:/Users/Admin/Desktop/DeepMoney_v2.0/model_dir/" + ver

#result file directory
result_dir = "C:/Users/Admin/Desktop/DeepMoney_v2.0/results_" + ver

#optimizing information file directory - hyperparameters
hyperpara_info_dir = "C:/Users/Admin/Desktop/DeepMoney_v2.0/hyperparameters/"

class Config(object):

    init_scale = 0.05
    learning_rate = 0.001
    num_layers = 2
    num_steps = 20
    hidden_size =500
    batch_size = 20
    input_size = 752
    output_size = 1
    rnn_mode = "basic"
    conversion = 'tot_norm'
    iter_steps = 1000
    step_interval = 20
    train_start =  "2000-01-01"
    test_start = "2017-07-01"
    test_end = "2018-09-30"
    predict_term = 65
    model_reset = True
    shuffle = True
