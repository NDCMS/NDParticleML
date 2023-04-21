#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nn_module as nnm
import torch
import numpy as np
import numpy.ma as ma
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import time
import copy
import pandas as pd
from pandas import read_csv
import argparse
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator


# In[2]:


out_file = '15708_2'
data_dict = {}
pp = PdfPages(f'./graphs/{out_file}_analysis.pdf')


# In[3]:


# The mapping between the WC names and their positions in the data
# There are Combine 1D scans over all 16 WCs
names = {}
names['cQei'] = 0
names['cQl3i'] = 1
names['cQlMi'] = 2
names['cbW'] = 3
names['cpQ3'] = 4
names['cpQM'] = 5
names['cpt'] = 6
names['cptb'] = 7
names['ctG'] = 8
names['ctW'] = 9
names['ctZ'] = 10
names['ctei'] = 11
names['ctlSi'] = 12
names['ctlTi'] = 13
names['ctli'] = 14
names['ctp'] = 15

# Pairs of WCs in existing Combine 2D scans
WC2d_1 = ['cpQ3', 'cpQM', 'cpt', 'cptb', 'cQei', 'ctei', 'ctli', 'ctW'] # The first of the two WCs to graph
WC2d_2 = ['cbW', 'ctG', 'ctp', 'cQl3i', 'cQlMi', 'ctlTi', 'ctlSi', 'ctZ'] # The second of the two WCs to graph

# Hyperparameters for profiling
profile_parameters = {}
profile_parameters['batch_size'] = 4096
profile_parameters['epochs'] = 100
profile_parameters['learning_rate'] = 2e-0
profile_parameters['lr_red_factor'] = 0.2
profile_parameters['lr_red_patience'] = 5
profile_parameters['lr_red_threshold'] = 1e-6
profile_parameters['rand_pts'] = 50 # The number of random starting points to do gradient descent on for each scanned value
profile_parameters['rand_stdev'] = 40

# general paremeters
batch_size = 4096 # for frozen graphs


# In[4]:


save_dict = torch.load(f'../models/{out_file}_model+.pt')
best_model_state = save_dict['model']
parameters_save = save_dict['parameters']
input_stats = save_dict['input_stats']
output_stats = save_dict['output_stats']


# In[5]:


# Check to make sure we're on cuda
input_stats[0].device


# In[6]:


# Check memory usage
torch.cuda.memory_allocated() / 1e9


# In[7]:


parameters_save


# In[8]:


best_model_state.keys()


# In[9]:


# Take Weinberg angle to be 30 degrees.
change_basis = torch.tensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,np.sqrt(3),1,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]]).float().cuda()

# Imagine this matrix applied to a vector. The first column of the matrix is what the first entry of the vector 
# gets mapped to, in this case a linear combination of the 10th and 11th WCs.


# In[10]:


model = nnm.create_model(16, 1, parameters_save, input_stats, output_stats, change_basis)
model.load_state_dict(best_model_state)
model.eval()


# In[11]:


# Data for 2D frozen graphs
# No comparison to Combine because this is new

model_2d_frozen_data = {}
# WCs to be scanned
idx2d_y = [9] # y axis
idx2d_x = [10] # x axis
res = 300 # How many points per axis
num_inputs = res*res

for idx in np.arange(len(idx2d_y)):
    inputs_y = np.linspace(-2,2,res)
    inputs_x = np.linspace(-7,7,res)
    # Generate the grid
    y_all, x_all = np.meshgrid(inputs_y, inputs_x)
    y_all = y_all.flatten()
    x_all = x_all.flatten()
    inputs = np.zeros((num_inputs, 16))
    inputs[:,idx2d_y[idx]] = y_all
    inputs[:,idx2d_x[idx]] = x_all
    inputs = torch.from_numpy(inputs).float().cuda()
    
    inputMiniBatches = torch.split(inputs, batch_size)
    outputs = torch.zeros((num_inputs, 1)).cuda()
    batch_idx = 0
    for minibatch in range(len(inputMiniBatches)):
        batch_inputs = inputMiniBatches[minibatch]
        this_batch_size = batch_inputs.shape[0]
        outputs[batch_idx: batch_idx + this_batch_size] = model(batch_inputs)
        batch_idx += this_batch_size
    
    outputs = outputs.cpu().detach().numpy().flatten()
    outputs *= 2
    # Save the x and y labels as well. These run from 0 to 15 to be consistent with the change of basis matrix.
    model_2d_frozen_data[str(idx)] = {'y': y_all, 'x': x_all, '2dNLL': outputs, 'y_label': f'x{idx2d_y[idx]}', 'x_label': f'x{idx2d_x[idx]}', 'change_basis': change_basis.cpu().detach().numpy()}


# In[12]:


# Save the data
data_dict['model_2d_frozen_data'] = model_2d_frozen_data


# In[13]:


# Data for 2D profiled graphs
# No comparison to Combine because this is new

model_2d_profiled_data = {}
# WCs to be scanned
# WARNING: Change names here and in graphing if you want to scan different WCs than for the frozen graphs
idx2d_y = [9] # y axis
idx2d_x = [10] # x axis
res = 300 # How many points per axis
num_inputs = res*res

for idx in np.arange(len(idx2d_y)):
    inputs_y = np.linspace(-2.5,2.5,res)
    inputs_x = np.linspace(-7.5,7.5,res)
    # Generate the grid
    y_all, x_all = np.meshgrid(inputs_y, inputs_x)
    y_all = y_all.flatten()
    x_all = x_all.flatten()
    inputs = np.zeros((num_inputs, 16))
    inputs[:,idx2d_y[idx]] = y_all
    inputs[:,idx2d_x[idx]] = x_all
    (min_WCs_scanned, outputs) = nnm.profile(model, inputs, [idx2d_y[idx], idx2d_x[idx]], profile_parameters)
    min_WCs_scanned = min_WCs_scanned.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy().flatten()
    outputs *= 2
    model_2d_profiled_data[str(idx)] = {'all_WCs': min_WCs_scanned, 'y': y_all, 'x': x_all, '2dNLL': outputs, 'y_label': f'x{idx2d_y[idx]}', 'x_label': f'x{idx2d_x[idx]}', 'change_basis': change_basis.cpu().detach().numpy()}


# In[14]:


# Save the data
data_dict['model_2d_profiled_data'] = model_2d_profiled_data


# In[15]:


# Make the graphs


# In[16]:


# Print the change of basis matrix first
change_basis_graph = plt.subplots()
#change_basis_graph[1].matshow(model_2d_frozen_data['0']['change_basis'], cmap=plt.cm.Blues)
for i in np.arange(change_basis.shape[0]):
    for j in np.arange(change_basis.shape[1]):
        c = model_2d_frozen_data['0']['change_basis'][j,i]
        change_basis_graph[1].text(i, j, f'{c:.3g}', va='center', ha='center', size=5)
change_basis_graph[1].set_xlim(-0.5, change_basis.shape[1]-0.5)
change_basis_graph[1].set_ylim(change_basis.shape[0]-0.5, -0.5)
change_basis_graph[1].set_xticks(np.arange(0, change_basis.shape[1], 1))
change_basis_graph[1].set_yticks(np.arange(0, change_basis.shape[0], 1))
change_basis_graph[1].xaxis.set_minor_locator(FixedLocator(np.arange(0.5, change_basis.shape[1]+1.5, 1)))
change_basis_graph[1].yaxis.set_minor_locator(FixedLocator(np.arange(0.5, change_basis.shape[0]+1.5, 1)))
change_basis_graph[1].grid(which='minor')
change_basis_graph[1].xaxis.set_tick_params(which='major', bottom=False)
change_basis_graph[1].yaxis.set_tick_params(which='major', bottom=False)
change_basis_graph[1].xaxis.set_tick_params(which='minor', bottom=False)
change_basis_graph[1].yaxis.set_tick_params(which='minor', bottom=False)
change_basis_graph[1].set_title('Change of Basis Matrix')
change_basis_graph[0].tight_layout()


# In[17]:


# Save the graphs
pp.savefig(change_basis_graph[0])


# In[18]:


# 2D frozen graphs
frozen_2d_graphs = {}
for idx in np.arange(len(idx2d_y)):
    model_2d_frozen_data[str(idx)]['2dNLL'] -= model_2d_frozen_data[str(idx)]['2dNLL'].min()
    frozen_2d_graphs[str(idx)] = plt.subplots()
    model_contour = frozen_2d_graphs[str(idx)][1].tricontour(model_2d_frozen_data[str(idx)]['x'], model_2d_frozen_data[str(idx)]['y'], model_2d_frozen_data[str(idx)]['2dNLL'], colors='r', linestyles=['dashed', 'dashdot', 'dotted'], levels=[2.30, 6.18, 11.83]) # 1, 2, and 3 sigmas
    SM_value = frozen_2d_graphs[str(idx)][1].scatter(0, 0, marker='d', c='gold', ec='royalblue', s=30, linewidths=1, zorder=10)
    frozen_2d_graphs[str(idx)][1].legend(model_contour.collections+[SM_value], ['$1\sigma$ predicted', '$2\sigma$ predicted', '$3\sigma$ predicted', 'SM value'])
    frozen_2d_graphs[str(idx)][1].set_xlabel(model_2d_frozen_data[str(idx)]['x_label'])
    frozen_2d_graphs[str(idx)][1].set_ylabel(model_2d_frozen_data[str(idx)]['y_label'])
    frozen_2d_graphs[str(idx)][1].set_title('Frozen')
    frozen_2d_graphs[str(idx)][0].tight_layout()


# In[19]:


# Save the graphs
for key in frozen_2d_graphs.keys():
    pp.savefig(frozen_2d_graphs[key][0])


# In[20]:


# 2D profiled graphs
profiled_2d_graphs = {}
for idx in np.arange(len(idx2d_y)):
    model_2d_profiled_data[str(idx)]['2dNLL'] -= model_2d_profiled_data[str(idx)]['2dNLL'].min()
    profiled_2d_graphs[str(idx)] = plt.subplots()
    model_contour = profiled_2d_graphs[str(idx)][1].tricontour(model_2d_profiled_data[str(idx)]['x'], model_2d_profiled_data[str(idx)]['y'], model_2d_profiled_data[str(idx)]['2dNLL'], colors='r', linestyles=['dashed', 'dashdot', 'dotted'], levels=[2.30, 6.18, 11.83]) # 1, 2, and 3 sigmas
    SM_value = profiled_2d_graphs[str(idx)][1].scatter(0, 0, marker='d', c='gold', ec='royalblue', s=30, linewidths=1, zorder=10)
    profiled_2d_graphs[str(idx)][1].legend(model_contour.collections+[SM_value], ['$1\sigma$ predicted', '$2\sigma$ predicted', '$3\sigma$ predicted', 'SM value'])
    profiled_2d_graphs[str(idx)][1].set_xlabel(model_2d_profiled_data[str(idx)]['x_label'])
    profiled_2d_graphs[str(idx)][1].set_ylabel(model_2d_profiled_data[str(idx)]['y_label'])
    profiled_2d_graphs[str(idx)][1].set_title('Profiled')
    profiled_2d_graphs[str(idx)][0].tight_layout()


# In[21]:


# Save the graphs
for key in profiled_2d_graphs.keys():
    pp.savefig(profiled_2d_graphs[key][0])


# In[22]:


# Save the graphs and data to files
np.savez(f'./graphs/{out_file}_analysis.npz', **data_dict)
pp.close()

