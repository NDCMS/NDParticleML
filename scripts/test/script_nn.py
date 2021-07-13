# %matplotlib notebook
import NN_Module as nnm
import torch
import numpy as np
import numpy.ma as ma
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import time
import pandas as pd
from pandas import read_csv
import argparse

# Functions

# Store arguments
parser = argparse.ArgumentParser(description='Train network.')
parser.add_argument('-o','--out-file', 
                    help='Name of output file')

parser.add_argument('-n','--nodes', 
                    type=int,
                    help='Number of nodes in each hidden layer')

parser.add_argument('-l','--layers', 
                    type=int,
                    help='Number of hidden layers')

parser.add_argument('-b','--batch-size',
                    type=int,
                    help='Minibatch size')

parser.add_argument('-e','--epochs',
                    type=int,
                    help='Number of epochs')

parser.add_argument('-lr','--learning-rate',
                    type=float,
                    help='Learning rate')

parser.add_argument('-lrr','--learning-rate-reduction',
                    type=float,
                    help='Learning rate reduction factor')

parser.add_argument('-lrrp','--learning-rate-patience',
                    type=float,
                    help='Learning rate reduction patience')

parser.add_argument('-lrrt','--learning-rate-threshold',
                    type=float,
                    help='Learning rate reduction threshold')

parser.add_argument('-wd','--weight-decay',
                    type=float,
                    help='Weight decay')

parser.add_argument('-aor','--accu-out-resolution',
                    type=int,
                    help='Resolution of the accu out graph')

parser.add_argument('-orr','--out-residual-resolution',
                    type=int,
                    help='Resolution of the out residual graph')

args = parser.parse_args()

# Load data sets
inputs = torch.load('inputs.pt').cuda()
outputs = torch.load('outputs.pt').cuda()
test_inputs = torch.load('test_inputs.pt').cuda()
test_outputs = torch.load('test_outputs.pt').cuda()

# Hyperparameters
parameters = {}
parameters['N'] = inputs.shape[1]
parameters['train_size'] = inputs.shape[0]
parameters['test_size'] = test_inputs.shape[0]
parameters['hidden_nodes'] = args.nodes
parameters['hidden_layers'] = args.layers
parameters['batch_size'] = args.batch_size
parameters['n_epochs'] = args.epochs
parameters['learning_rate'] = args.learning_rate
parameters['lr_red_factor'] = args.learning_rate_reduction
parameters['lr_red_patience'] = args.learning_rate_patience
parameters['lr_red_threshold'] = args.learning_rate_threshold
parameters['weight_decay'] = args.weight_decay

# Graphing parameters
parameters['accu_out_resolution'] = args.accu_out_resolution
parameters['out_residual_resolution'] = args.out_residual_resolution

# Standardize data sets
input_stats = nnm.find_stats(inputs)
output_stats = nnm.find_stats(outputs)
std_inputs = nnm.affine_transform(inputs, input_stats)
std_test_inputs = nnm.affine_transform(test_inputs, input_stats) # Not actually normal; Only std_inputs is normal
std_outputs = nnm.affine_transform(outputs, output_stats)
std_test_outputs = nnm.affine_transform(test_outputs, output_stats)

# Create a representative training set
index_std_train_rep = np.random.choice(parameters['train_size'], parameters['test_size'], replace=False)
std_inputs_rep = std_inputs[index_std_train_rep]
std_outputs_rep = std_outputs[index_std_train_rep]

# Create a model
model = nnm.create_model(inputs, outputs, parameters)

# Train the model
(graph_data, best_model_state, parameters_save) = nnm.train_network(model, std_inputs, std_outputs, std_test_inputs, std_test_outputs, output_stats, std_inputs_rep, std_outputs_rep, parameters)

# Graphing
graphs = nnm.new_graphs()
nnm.graphing(graphs, graph_data, parameters)

# Save all graphs
nnm.save_graphs(graphs, f'./graphs/{args.out_file}.pdf')

# See pytorch version
# print (torch.__version__)