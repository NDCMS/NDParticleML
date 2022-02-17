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
parser.add_argument('-m','--min', 
                    type=int,
                    help='Minimum output to be included in the training set')
                    
parser.add_argument('-M','--max', 
                    type=int,
                    help='Maximum output to be included in the training set')

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
loaded_data = np.load('likelihood_data_no_delta.npz')
use_it = (loaded_data['deltaNLL'] != 0)
use_it[0] = True # Keep the first of the repeated entries
outputs_all = loaded_data['deltaNLL'][use_it]
inputs_all = []
inputs_all.append(loaded_data['cQei'][use_it])
inputs_all.append(loaded_data['cQl3i'][use_it])
inputs_all.append(loaded_data['cQlMi'][use_it])
inputs_all.append(loaded_data['cbW'][use_it])
inputs_all.append(loaded_data['cpQ3'][use_it])
inputs_all.append(loaded_data['cpQM'][use_it])
inputs_all.append(loaded_data['cpt'][use_it])
inputs_all.append(loaded_data['cptb'][use_it])
inputs_all.append(loaded_data['ctG'][use_it])
inputs_all.append(loaded_data['ctW'][use_it])
inputs_all.append(loaded_data['ctZ'][use_it])
inputs_all.append(loaded_data['ctei'][use_it])
inputs_all.append(loaded_data['ctlSi'][use_it])
inputs_all.append(loaded_data['ctlTi'][use_it])
inputs_all.append(loaded_data['ctli'][use_it])
inputs_all.append(loaded_data['ctp'][use_it])
inputs_all = np.stack(inputs_all, axis=1)
inmin = np.min(inputs_all)
inputs_all = inputs_all-inmin
inputs_all = inputs_all[:5*10**5,:]
outputs_all = outputs_all[:5*10**5]
# Randomize
all = np.concatenate([inputs_all, np.expand_dims(outputs_all, axis=1)], axis=1)
np.random.shuffle(all)
inputs_all = all[:,:-1]
outputs_all = all[:,-1]

# Add the squares of the variables and cross terms
inputs_list = []
inputs_list.append(inputs_all)
inputs_all_squared = inputs_all ** 2
inputs_list.append(inputs_all_squared)
# Add the cross terms
for i in range(16):
    for j in range(i):
        inputs_list.append(np.expand_dims(inputs_all[:,i] *  inputs_all[:,j], axis=1))
inputs_all = np.concatenate(inputs_list, axis=1)

# Save only the points in the interval; don't forget to change the size of the network accordingly!
lower = args.min
upper = args.max
idx = (outputs_all > lower) & (outputs_all < upper)
outputs_all = outputs_all[idx]
inputs_all = inputs_all[idx]

# Prepare to split into training and validation sets
total_data = outputs_all.shape[0]
train_proportion = 0.99
validation_proportion = 0.01

# Create a representative validation set
index_test = np.random.choice(total_data, int(total_data*validation_proportion), replace=False)
test_inputs = inputs_all[index_test]
test_outputs = outputs_all[index_test]
inputs = np.delete(inputs_all, index_test, axis=0)
outputs = np.delete(outputs_all, index_test, axis=0)

inputs = torch.from_numpy(inputs).cuda()
outputs = torch.unsqueeze(torch.from_numpy(outputs).cuda(), 1)
test_inputs = torch.from_numpy(test_inputs).cuda()
test_outputs = torch.unsqueeze(torch.from_numpy(test_outputs).cuda(), 1)

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
print(torch.cuda.memory_summary(device=None, abbreviated=False))
# Create a model
model = nnm.create_model(inputs.shape[1], outputs.shape[1], parameters)
print(torch.cuda.memory_summary(device=None, abbreviated=False))
# Train the model
(graph_data, best_model_state, parameters_save) = nnm.train_network(model, std_inputs, std_outputs, std_test_inputs, std_test_outputs, output_stats, std_inputs_rep, std_outputs_rep, parameters)

# Graphing
graphs = nnm.new_graphs()
nnm.graphing(graphs, graph_data, parameters)

# Save all graphs
nnm.save_graphs(graphs, f'./graphs/{args.out_file}.pdf')

# Save the model, parameters, and standardization stats
save_dict = {'model': best_model_state, 'parameters': parameters_save, 'input_stats': input_stats, 'output_stats': output_stats}
torch.save(save_dict, f'./models/{args.out_file}_model+.pt')

# See pytorch version
# print (torch.__version__)
