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

parser.add_argument('-mnout','--min-output',
                    type=float,
                    help='Minimum desired output range')

parser.add_argument('-mxout','--max-output',
                    type=float,
                    help='Maximum desired output range')

args = parser.parse_args()

# Load data sets
loaded_data = np.load('/scratch365/klannon/dnnlikelihood/likelihood_data_no_delta.npz')
deltaNLL = loaded_data['deltaNLL']-np.min(loaded_data['deltaNLL'])

# Tune proportion of outputs over 50 vs under 50
use_it_over = (deltaNLL >= 50) # Only keep the outputs > 50
use_it_under = (deltaNLL < 50) 
under_50_num = deltaNLL[use_it_under].size
over_50_num = deltaNLL[use_it_over].size
under_proportion = 1     # For now, these are set to use all the data as it is (without tweaks to proportion of over/under 50)
over_proportion = 1
index_under = np.random.choice(under_50_num, int(under_50_num*under_proportion), replace=False)
index_over = np.random.choice(over_50_num, int(over_50_num*over_proportion), replace=False)

# Load desired proportion of WC values
cQei = np.concatenate((loaded_data['cQei'][index_under], loaded_data['cQei'][index_over]), axis=0)
cQl3i = np.concatenate((loaded_data['cQl3i'][index_under], loaded_data['cQl3i'][index_over]), axis=0)
cQlMi = np.concatenate((loaded_data['cQlMi'][index_under], loaded_data['cQlMi'][index_over]), axis=0)
cbW = np.concatenate((loaded_data['cbW'][index_under], loaded_data['cbW'][index_over]), axis=0)
cpQ3 = np.concatenate((loaded_data['cpQ3'][index_under], loaded_data['cpQ3'][index_over]), axis=0)
cpQM = np.concatenate((loaded_data['cpQM'][index_under], loaded_data['cpQM'][index_over]), axis=0)
cpt = np.concatenate((loaded_data['cpt'][index_under], loaded_data['cpt'][index_over]), axis=0)
cptb = np.concatenate((loaded_data['cptb'][index_under], loaded_data['cptb'][index_over]), axis=0)
ctG = np.concatenate((loaded_data['ctG'][index_under], loaded_data['ctG'][index_over]), axis=0)
ctW = np.concatenate((loaded_data['ctW'][index_under], loaded_data['ctW'][index_over]), axis=0)
ctZ = np.concatenate((loaded_data['ctZ'][index_under], loaded_data['ctZ'][index_over]), axis=0)
ctei = np.concatenate((loaded_data['ctei'][index_under], loaded_data['ctei'][index_over]), axis=0)
ctlSi = np.concatenate((loaded_data['ctlSi'][index_under], loaded_data['ctlSi'][index_over]), axis=0)
ctlTi = np.concatenate((loaded_data['ctlTi'][index_under], loaded_data['ctlTi'][index_over]), axis=0)
ctli = np.concatenate((loaded_data['ctli'][index_under], loaded_data['ctli'][index_over]), axis=0)
ctp = np.concatenate((loaded_data['ctp'][index_under], loaded_data['ctp'][index_over]), axis=0)

# Load desired proportion of outputs
outputs_all = np.concatenate((deltaNLL[index_under], deltaNLL[index_over]), axis=0)

# Store values in list
inputs_all = []
inputs_all.append(cQei)
inputs_all.append(cQl3i)
inputs_all.append(cQlMi)
inputs_all.append(cbW)
inputs_all.append(cpQ3)
inputs_all.append(cpQM)
inputs_all.append(cpt)
inputs_all.append(cptb)
inputs_all.append(ctG)
inputs_all.append(ctW)
inputs_all.append(ctZ)
inputs_all.append(ctei)
inputs_all.append(ctlSi)
inputs_all.append(ctlTi)
inputs_all.append(ctli)
inputs_all.append(ctp)
inputs_all = np.stack(inputs_all, axis=1)

# Randomize
all = np.concatenate([inputs_all, np.expand_dims(outputs_all, axis=1)], axis=1)
np.random.shuffle(all)
inputs_all = all[:,:-1]
outputs_all = all[:,-1]

# # Take desired fraction of data (for memory purposes)
# total_data = loaded_data['deltaNLL'].size
# use_proportion = 0.16
# index_data = np.random.choice(total_data, int(total_data*use_proportion), replace=False)
# inputs_all = inputs_all[index_data]
# outputs_all = outputs_all[index_data]

# Add the squares of the variables and cross terms
inputs_list = []
inputs_list.append(inputs_all)
# inputs_all_squared = inputs_all ** 2
# inputs_list.append(inputs_all_squared)

# Add the cross terms
# for i in range(16):
#     for j in range(i):
#         inputs_list.append(np.expand_dims(inputs_all[:,i] *  inputs_all[:,j], axis=1))
# inputs_all = np.concatenate(inputs_list, axis=1)

# Save only the points with output in given range
min_out = args.min_output
max_out = args.max_output
use_range = (outputs_all >= min_out) & (outputs_all < max_out)
outputs_all = outputs_all[use_range]
inputs_all = inputs_all[use_range]

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

# Create a model
model = nnm.create_model(inputs.shape[1], outputs.shape[1], parameters)

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
