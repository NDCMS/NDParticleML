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

out_file = '15708_2'
data_dict = {}
pp = PdfPages(f'./graphs/{out_file}_validation.pdf')

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

save_dict = torch.load(f'./{out_file}_model+.pt')
best_model_state = save_dict['model']
parameters_save = save_dict['parameters']
input_stats = save_dict['input_stats']
output_stats = save_dict['output_stats']

# Check to make sure we're on cuda
print (f'Current device: {input_stats[0].device}')

# Check memory usage
print (f'Memory usage: {torch.cuda.memory_allocated() / 1e9} GB')

# TODO: Cheat a little. Fix later by training with the additional layers.
'''
parameters_save['polynomial'] = True
parameters_save['polynomial_degree'] = 2
best_model_state['1.linear.weight'] = best_model_state.pop('0.linear.weight')
best_model_state['1.linear.bias'] = best_model_state.pop('0.linear.bias')
best_model_state['3.weight'] = best_model_state.pop('2.weight')
best_model_state['3.bias'] = best_model_state.pop('2.bias')
best_model_state['5.weight'] = best_model_state.pop('4.weight')
best_model_state['5.bias'] = best_model_state.pop('4.bias')
best_model_state['7.weight'] = best_model_state.pop('6.weight')
best_model_state['7.bias'] = best_model_state.pop('6.bias')
'''

model = nnm.create_model(16, 1, parameters_save, input_stats, output_stats)
model.load_state_dict(best_model_state)
model.eval()

# Data for 1D frozen graphs compared to Combine scans
target_1d_frozen_data = {}
for key in names.keys():
    loaded = np.load(f'likelihood_{key}.npz')
    target_1d_frozen_data[key] = {key: loaded[key], '2dNLL': loaded['deltaNLL']}
    target_1d_frozen_data[key]['2dNLL'] *= 2

target_1d_zoomed_frozen_data = {}
for key in names.keys():
    less_than_10 = (target_1d_frozen_data[key]['2dNLL'] < 10)
    target_1d_zoomed_frozen_data[key] = {key: target_1d_frozen_data[key][key][less_than_10], '2dNLL': target_1d_frozen_data[key]['2dNLL'][less_than_10]}

model_1d_frozen_data = {}
diff_1d_frozen_data = {}
for key in names.keys():
    inputs = target_1d_frozen_data[key][key]
    num_inputs = inputs.shape[0]
    # Now requires model to have been trained with the polynomial layer
    if (parameters_save['polynomial']):
        inputs_all = np.zeros((num_inputs, 16))
        inputs_all[:,names[key]] = inputs
    else:
        raise RuntimeError(f'Models without the polynomial layer are no longer supported! The number of inputs must match the number of WCs.')
    outputs = model(torch.from_numpy(inputs_all).float().cuda()).cpu().detach().numpy().flatten()
    outputs *= 2
    model_1d_frozen_data[key] = {key: inputs, '2dNLL': outputs}
    diff_1d_frozen_data[key] = {key: inputs, '2dNLL': target_1d_frozen_data[key]['2dNLL'] - model_1d_frozen_data[key]['2dNLL']}
    
model_1d_zoomed_frozen_data = {}
diff_1d_zoomed_frozen_data = {}
for key in names.keys():
    less_than_10 = (target_1d_frozen_data[key]['2dNLL'] < 10)
    model_1d_zoomed_frozen_data[key] = {key: model_1d_frozen_data[key][key][less_than_10], '2dNLL': model_1d_frozen_data[key]['2dNLL'][less_than_10]}
    diff_1d_zoomed_frozen_data[key] = {key: diff_1d_frozen_data[key][key][less_than_10], '2dNLL': diff_1d_frozen_data[key]['2dNLL'][less_than_10]}

# Save the data
data_dict['target_1d_frozen_data'] = target_1d_frozen_data
data_dict['model_1d_frozen_data'] = model_1d_frozen_data
data_dict['diff_1d_frozen_data'] = diff_1d_frozen_data

# Data for 1D fake profiled graphs compared to Combine scans
# Target data
target_1d_profiled_data = {}
for key in names.keys():
    loaded = np.load(f'likelihood_profiled_{key}.npz')
    inputs = np.zeros((loaded['deltaNLL'].shape[0], 16))
    for key2 in names.keys():
        inputs[:,names[key2]] = loaded[key2]
    target_1d_profiled_data[key] = {'all_WCs': inputs, '2dNLL': loaded['deltaNLL']} # Inputs here contain all the WCs
    target_1d_profiled_data[key]['2dNLL'] *= 2
    
target_1d_zoomed_profiled_data = {}
for key in names.keys():
    less_than_10 = (target_1d_profiled_data[key]['2dNLL'] < 10)
    target_1d_zoomed_profiled_data[key] = {'all_WCs': target_1d_profiled_data[key]['all_WCs'][less_than_10], '2dNLL': target_1d_profiled_data[key]['2dNLL'][less_than_10]}

# Save the data
data_dict['target_1d_profiled_data'] = target_1d_profiled_data

# Model and diff data
model_1d_fake_profiled_data = {}
diff_1d_fake_profiled_data = {}
for key in names.keys():
    inputs = target_1d_profiled_data[key]['all_WCs']
    # Now requires model to have been trained with the polynomial layer
    if not (parameters_save['polynomial']):
        raise RuntimeError(f'Models without the polynomial layer are no longer supported! The number of inputs must match the number of WCs.')
    outputs = model(torch.from_numpy(inputs).float().cuda()).cpu().detach().numpy().flatten()
    outputs *= 2
    model_1d_fake_profiled_data[key] = {'all_WCs': inputs, '2dNLL': outputs}
    diff_1d_fake_profiled_data[key] = {'all_WCs': inputs, '2dNLL': target_1d_profiled_data[key]['2dNLL'] - outputs}
    
model_1d_zoomed_fake_profiled_data = {}
diff_1d_zoomed_fake_profiled_data = {}
for key in names.keys():
    less_than_10 = (target_1d_profiled_data[key]['2dNLL'] < 10)
    model_1d_zoomed_fake_profiled_data[key] = {'all_WCs': model_1d_fake_profiled_data[key]['all_WCs'][less_than_10], '2dNLL': model_1d_fake_profiled_data[key]['2dNLL'][less_than_10]}
    diff_1d_zoomed_fake_profiled_data[key] = {'all_WCs': diff_1d_fake_profiled_data[key]['all_WCs'][less_than_10], '2dNLL': diff_1d_fake_profiled_data[key]['2dNLL'][less_than_10]}

# Save the data
data_dict['model_1d_fake_profiled_data'] = model_1d_fake_profiled_data
data_dict['diff_1d_fake_profiled_data'] = diff_1d_fake_profiled_data

# Data for 2D frozen graphs compared to Combine scans
target_2d_frozen_data = {}
for num in np.arange(8):
    WC1 = WC2d_1[num]
    WC2 = WC2d_2[num]
    loaded = np.load(f'likelihood_{WC1}_{WC2}.npz')
    target_2d_frozen_data[str(num)] = {WC1: loaded[WC1], WC2: loaded[WC2], '2dNLL': loaded['deltaNLL']}
    target_2d_frozen_data[str(num)]['2dNLL'] *= 2

model_2d_frozen_data = {}
diff_2d_frozen_data = {}
for num in np.arange(8):
    inputs_y = target_2d_frozen_data[str(num)][WC2d_1[num]]
    inputs_x = target_2d_frozen_data[str(num)][WC2d_2[num]]
    num_inputs = inputs_y.shape[0]
    inputs = np.zeros((num_inputs, 16))
    inputs[:,names[WC2d_1[num]]] = inputs_y
    inputs[:,names[WC2d_2[num]]] = inputs_x
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
    model_2d_frozen_data[str(num)] = {WC2d_1[num]: target_2d_frozen_data[str(num)][WC2d_1[num]], WC2d_2[num]: target_2d_frozen_data[str(num)][WC2d_2[num]], '2dNLL': outputs}
    diff_2d_frozen_data[str(num)] = {WC2d_1[num]: target_2d_frozen_data[str(num)][WC2d_1[num]], WC2d_2[num]: target_2d_frozen_data[str(num)][WC2d_2[num]], '2dNLL': target_2d_frozen_data[str(num)]['2dNLL'] - outputs}

# Save the data
data_dict['target_2d_frozen_data'] = target_2d_frozen_data
data_dict['model_2d_frozen_data'] = model_2d_frozen_data
data_dict['diff_2d_frozen_data'] = diff_2d_frozen_data

# Data for 2D fake profiled graphs compared to Combine scans
# Target data
target_2d_profiled_data = {}
for num in np.arange(8):
    WC1 = WC2d_1[num]
    WC2 = WC2d_2[num]
    loaded = np.load(f'likelihood_profiled_{WC1}_{WC2}.npz')
    inputs = np.zeros((loaded['deltaNLL'].shape[0], 16))
    for key2 in names.keys():
        inputs[:,names[key2]] = loaded[key2]
    target_2d_profiled_data[str(num)] = {'all_WCs': inputs, '2dNLL': loaded['deltaNLL']} # Inputs here contain all the WCs
    target_2d_profiled_data[str(num)]['2dNLL'] *= 2

# Save the data
data_dict['target_2d_profiled_data'] = target_2d_profiled_data

# Model data
model_2d_fake_profiled_data = {}
for num in np.arange(8):
    inputs = target_2d_profiled_data[str(num)]['all_WCs']
    num_inputs = inputs.shape[0]
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
    model_2d_fake_profiled_data[str(num)] = {'all_WCs': target_2d_profiled_data[str(num)]['all_WCs'], '2dNLL': outputs}

# Save the data
data_dict['model_2d_fake_profiled_data'] = model_2d_fake_profiled_data

# Data for 1D real profiled graphs compared to Combine scans
# For target data see before
# Model and diff data
model_1d_profiled_data = {}
diff_1d_profiled_data = {}
for key in names.keys():
    inputs = target_1d_profiled_data[key]['all_WCs']
    # Now requires model to have been trained with the polynomial layer
    if not (parameters_save['polynomial']):
        raise RuntimeError(f'Models without the polynomial layer are no longer supported! The number of inputs must match the number of WCs.')
    # profile() currently takes inputs as an np array not tensor. I know, confusing, so to be fixed in the future.
    (min_WCs_scanned, outputs) = nnm.profile(model, inputs, [names[key]], profile_parameters)
    min_WCs_scanned = min_WCs_scanned.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy().flatten()
    outputs *= 2
    model_1d_profiled_data[key] = {'all_WCs': min_WCs_scanned, '2dNLL': outputs}
    diff_1d_profiled_data[key] = {'all_WCs': target_1d_profiled_data[key]['all_WCs'], '2dNLL': target_1d_profiled_data[key]['2dNLL'] - outputs}
    
model_1d_zoomed_profiled_data = {}
diff_1d_zoomed_profiled_data = {}
for key in names.keys():
    less_than_10 = (target_1d_profiled_data[key]['2dNLL'] < 10)
    model_1d_zoomed_profiled_data[key] = {'all_WCs': model_1d_profiled_data[key]['all_WCs'][less_than_10], '2dNLL': model_1d_profiled_data[key]['2dNLL'][less_than_10]}
    diff_1d_zoomed_profiled_data[key] = {'all_WCs': diff_1d_profiled_data[key]['all_WCs'][less_than_10], '2dNLL': diff_1d_profiled_data[key]['2dNLL'][less_than_10]}

# Save the data
data_dict['model_1d_profiled_data'] = model_1d_profiled_data
data_dict['diff_1d_profiled_data'] = diff_1d_profiled_data

# Data for 2D real profiled graphs compared to Combine scans
# Model data
model_2d_profiled_data = {}
# Because this takes some time, select which graphs you want to make.
nums = [0,1,2,3,4,5,6,7]
for num in nums:
    inputs = target_2d_profiled_data[str(num)]['all_WCs']
    (min_WCs_scanned, outputs) = nnm.profile(model, inputs, [names[WC2d_1[num]], names[WC2d_2[num]]], profile_parameters)
    min_WCs_scanned = min_WCs_scanned.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy().flatten()
    outputs *= 2
    model_2d_profiled_data[str(num)] = {'all_WCs': min_WCs_scanned, '2dNLL': outputs}

# Save the data
data_dict['model_2d_profiled_data'] = model_2d_profiled_data

# Make the graphs

# 1D frozen graphs compared to Combine scans
frozen_1d_graphs = {}
for key in names.keys():
    target_1d_frozen_data[key]['2dNLL'] -= target_1d_frozen_data[key]['2dNLL'].min()
    model_1d_frozen_data[key]['2dNLL'] -= model_1d_frozen_data[key]['2dNLL'].min()
    frozen_1d_graphs[key] = plt.subplots()
    target_scatter = frozen_1d_graphs[key][1].scatter(target_1d_frozen_data[key][key], target_1d_frozen_data[key]['2dNLL'], marker='^', c='none', ec='k', s=5, linewidths=0.2)
    model_scatter = frozen_1d_graphs[key][1].scatter(model_1d_frozen_data[key][key], model_1d_frozen_data[key]['2dNLL'], marker='v', c='none', ec='g', s=5, linewidths=0.2)
    one_line, = frozen_1d_graphs[key][1].plot([target_1d_frozen_data[key][key].min(), target_1d_frozen_data[key][key].max()], [1,1], 'r--', linewidth=0.5)
    four_line, = frozen_1d_graphs[key][1].plot([target_1d_frozen_data[key][key].min(), target_1d_frozen_data[key][key].max()], [4,4], 'm:', linewidth=0.5)
    frozen_1d_graphs[key][1].legend([target_scatter, model_scatter, one_line, four_line], ['Target', 'NN Prediction', '$1\sigma\ (2\Delta NLL=1)$', '$2\sigma\ (2\Delta NLL=4)$'], markerscale=3)
    frozen_1d_graphs[key][1].set_xlabel(key)
    frozen_1d_graphs[key][1].set_ylabel('2$\Delta$NLL')
    frozen_1d_graphs[key][1].set_title('Frozen')
    frozen_1d_graphs[key][0].tight_layout()

'''
diff_frozen_1d_graphs = {}
for key in names.keys():
    diff_frozen_1d_graphs[key] = plt.subplots()
    diff_frozen_1d_graphs[key][1].scatter(diff_1d_frozen_data[key][key], diff_1d_frozen_data[key]['2dNLL'], c='b', s=1)
    diff_frozen_1d_graphs[key][1].set_xlabel(key)
    diff_frozen_1d_graphs[key][1].set_ylabel('residual (target - prediction)')
    diff_frozen_1d_graphs[key][1].set_title('Frozen')
    diff_frozen_1d_graphs[key][0].tight_layout()
'''
    
zoomed_frozen_1d_graphs = {}
for key in names.keys():
    target_1d_zoomed_frozen_data[key]['2dNLL'] -= target_1d_zoomed_frozen_data[key]['2dNLL'].min()
    model_1d_zoomed_frozen_data[key]['2dNLL'] -= model_1d_zoomed_frozen_data[key]['2dNLL'].min()
    zoomed_frozen_1d_graphs[key] = plt.subplots()
    target_scatter = zoomed_frozen_1d_graphs[key][1].scatter(target_1d_zoomed_frozen_data[key][key], target_1d_zoomed_frozen_data[key]['2dNLL'], marker='^', c='none', ec='k', s=5, linewidths=0.2)
    model_scatter = zoomed_frozen_1d_graphs[key][1].scatter(model_1d_zoomed_frozen_data[key][key], model_1d_zoomed_frozen_data[key]['2dNLL'], marker='v', c='none', ec='g', s=5, linewidths=0.2)
    one_line, = zoomed_frozen_1d_graphs[key][1].plot([target_1d_zoomed_frozen_data[key][key].min(), target_1d_zoomed_frozen_data[key][key].max()], [1,1], 'r--', linewidth=0.5)
    four_line, = zoomed_frozen_1d_graphs[key][1].plot([target_1d_zoomed_frozen_data[key][key].min(), target_1d_zoomed_frozen_data[key][key].max()], [4,4], 'm:', linewidth=0.5)
    zoomed_frozen_1d_graphs[key][1].legend([target_scatter, model_scatter, one_line, four_line], ['Target', 'NN Prediction', '$1\sigma\ (2\Delta NLL=1)$', '$2\sigma\ (2\Delta NLL=4)$'], markerscale=3)
    zoomed_frozen_1d_graphs[key][1].set_xlabel(key)
    zoomed_frozen_1d_graphs[key][1].set_ylabel('2$\Delta$NLL')
    zoomed_frozen_1d_graphs[key][1].set_title('Frozen')
    zoomed_frozen_1d_graphs[key][0].tight_layout()

# Save the graphs
for key in frozen_1d_graphs.keys():
    pp.savefig(frozen_1d_graphs[key][0])
'''
for key in diff_frozen_1d_graphs.keys():
    pp.savefig(diff_frozen_1d_graphs[key][0])
'''
for key in zoomed_frozen_1d_graphs.keys():
    pp.savefig(zoomed_frozen_1d_graphs[key][0])

# 1D fake profiled graphs compared to Combine scans
fake_profiled_1d_graphs = {}
for key in names.keys():
    target_1d_profiled_data[key]['2dNLL'] -= target_1d_profiled_data[key]['2dNLL'].min()
    model_1d_fake_profiled_data[key]['2dNLL'] -= model_1d_fake_profiled_data[key]['2dNLL'].min()
    fake_profiled_1d_graphs[key] = plt.subplots()
    target_scatter = fake_profiled_1d_graphs[key][1].scatter(target_1d_profiled_data[key]['all_WCs'][:,names[key]], target_1d_profiled_data[key]['2dNLL'], marker='^', c='none', ec='k', s=5, linewidths=0.2)
    model_scatter = fake_profiled_1d_graphs[key][1].scatter(model_1d_fake_profiled_data[key]['all_WCs'][:,names[key]], model_1d_fake_profiled_data[key]['2dNLL'], marker='v', c='none', ec='g', s=5, linewidths=0.2)
    one_line, = fake_profiled_1d_graphs[key][1].plot([target_1d_profiled_data[key]['all_WCs'][:,names[key]].min(), target_1d_profiled_data[key]['all_WCs'][:,names[key]].max()], [1,1], 'r--', linewidth=0.5)
    four_line, = fake_profiled_1d_graphs[key][1].plot([target_1d_profiled_data[key]['all_WCs'][:,names[key]].min(), target_1d_profiled_data[key]['all_WCs'][:,names[key]].max()], [4,4], 'm:', linewidth=0.5)
    fake_profiled_1d_graphs[key][1].legend([target_scatter, model_scatter, one_line, four_line], ['Target', 'NN Prediction', '$1\sigma\ (2\Delta NLL=1)$', '$2\sigma\ (2\Delta NLL=4)$'], markerscale=3)
    fake_profiled_1d_graphs[key][1].set_xlabel(key)
    fake_profiled_1d_graphs[key][1].set_ylabel('2$\Delta$NLL')
    fake_profiled_1d_graphs[key][1].set_title('Fake Profiled')
    fake_profiled_1d_graphs[key][0].tight_layout()

zoomed_fake_profiled_1d_graphs = {}
for key in names.keys():
    target_1d_zoomed_profiled_data[key]['2dNLL'] -= target_1d_zoomed_profiled_data[key]['2dNLL'].min()
    model_1d_zoomed_fake_profiled_data[key]['2dNLL'] -= model_1d_zoomed_fake_profiled_data[key]['2dNLL'].min()
    zoomed_fake_profiled_1d_graphs[key] = plt.subplots()
    target_scatter = zoomed_fake_profiled_1d_graphs[key][1].scatter(target_1d_zoomed_profiled_data[key]['all_WCs'][:,names[key]], target_1d_zoomed_profiled_data[key]['2dNLL'], marker='^', c='none', ec='k', s=5, linewidths=0.2)
    model_scatter = zoomed_fake_profiled_1d_graphs[key][1].scatter(model_1d_zoomed_fake_profiled_data[key]['all_WCs'][:,names[key]], model_1d_zoomed_fake_profiled_data[key]['2dNLL'], marker='v', c='none', ec='g', s=5, linewidths=0.2)
    one_line, = zoomed_fake_profiled_1d_graphs[key][1].plot([target_1d_zoomed_profiled_data[key]['all_WCs'][:,names[key]].min(), target_1d_zoomed_profiled_data[key]['all_WCs'][:,names[key]].max()], [1,1], 'r--', linewidth=0.5)
    four_line, = zoomed_fake_profiled_1d_graphs[key][1].plot([target_1d_zoomed_profiled_data[key]['all_WCs'][:,names[key]].min(), target_1d_zoomed_profiled_data[key]['all_WCs'][:,names[key]].max()], [4,4], 'm:', linewidth=0.5)
    zoomed_fake_profiled_1d_graphs[key][1].legend([target_scatter, model_scatter, one_line, four_line], ['Target', 'NN Prediction', '$1\sigma\ (2\Delta NLL=1)$', '$2\sigma\ (2\Delta NLL=4)$'], markerscale=3)
    zoomed_fake_profiled_1d_graphs[key][1].set_xlabel(key)
    zoomed_fake_profiled_1d_graphs[key][1].set_ylabel('2$\Delta$NLL')
    zoomed_fake_profiled_1d_graphs[key][1].set_title('Fake Profiled')
    zoomed_fake_profiled_1d_graphs[key][0].tight_layout()

# Save the graphs
for key in fake_profiled_1d_graphs.keys():
    pp.savefig(fake_profiled_1d_graphs[key][0])
for key in zoomed_fake_profiled_1d_graphs.keys():
    pp.savefig(zoomed_fake_profiled_1d_graphs[key][0])

# 2D frozen graphs compared to Combine scans
frozen_2d_graphs = {}
for num in np.arange(8):
    target_2d_frozen_data[str(num)]['2dNLL'] -= target_2d_frozen_data[str(num)]['2dNLL'].min()
    model_2d_frozen_data[str(num)]['2dNLL'] -= model_2d_frozen_data[str(num)]['2dNLL'].min()
    frozen_2d_graphs[str(num)] = plt.subplots()
    target_contour = frozen_2d_graphs[str(num)][1].tricontour(target_2d_frozen_data[str(num)][WC2d_2[num]], target_2d_frozen_data[str(num)][WC2d_1[num]], target_2d_frozen_data[str(num)]['2dNLL'], colors='k', linestyles=['dashed', 'dashdot', 'dotted'], levels=[2.30, 6.18, 11.83]) # 1, 2, and 3 sigmas
    model_contour = frozen_2d_graphs[str(num)][1].tricontour(model_2d_frozen_data[str(num)][WC2d_2[num]], model_2d_frozen_data[str(num)][WC2d_1[num]], model_2d_frozen_data[str(num)]['2dNLL'], colors='r', linestyles=['dashed', 'dashdot', 'dotted'], levels=[2.30, 6.18, 11.83]) # 1, 2, and 3 sigmas
    SM_value = frozen_2d_graphs[str(num)][1].scatter(0, 0, marker='d', c='gold', ec='royalblue', s=30, linewidths=1, zorder=10)
    frozen_2d_graphs[str(num)][1].legend(target_contour.collections+model_contour.collections+[SM_value], ['$1\sigma$ target', '$2\sigma$ target', '$3\sigma$ target', '$1\sigma$ predicted', '$2\sigma$ predicted', '$3\sigma$ predicted', 'SM value'])
    frozen_2d_graphs[str(num)][1].set_xlabel(WC2d_2[num])
    frozen_2d_graphs[str(num)][1].set_ylabel(WC2d_1[num])
    frozen_2d_graphs[str(num)][1].set_title('Frozen')
    frozen_2d_graphs[str(num)][0].tight_layout()

'''
diff_frozen_2d_graphs = {}
for num in np.arange(8):
    diff_frozen_2d_graphs[str(num)] = plt.subplots()
    im = diff_frozen_2d_graphs[str(num)][1].tripcolor(diff_2d_frozen_data[str(num)][WC2d_2[num]], diff_2d_frozen_data[str(num)][WC2d_1[num]], diff_2d_frozen_data[str(num)]['2dNLL'])
    diff_frozen_2d_graphs[str(num)][1].set_xlabel(WC2d_2[num])
    diff_frozen_2d_graphs[str(num)][1].set_ylabel(WC2d_1[num])
    diff_frozen_2d_graphs[str(num)][1].set_title('Frozen')
    diff_frozen_2d_graphs[str(num)][0].colorbar(im, ax=diff_frozen_2d_graphs[str(num)][1], label='target - prediction')
    diff_frozen_2d_graphs[str(num)][0].tight_layout()
'''

# Save the graphs
for key in frozen_2d_graphs.keys():
    pp.savefig(frozen_2d_graphs[key][0])
'''
for key in diff_frozen_2d_graphs.keys():
    pp.savefig(diff_frozen_2d_graphs[key][0])
'''

# 2D fake profiled graphs compared to Combine scans
fake_profiled_2d_graphs = {}
for num in np.arange(8):
    target_2d_profiled_data[str(num)]['2dNLL'] -= target_2d_profiled_data[str(num)]['2dNLL'].min()
    model_2d_fake_profiled_data[str(num)]['2dNLL'] -= model_2d_fake_profiled_data[str(num)]['2dNLL'].min()
    fake_profiled_2d_graphs[str(num)] = plt.subplots()
    target_contour = fake_profiled_2d_graphs[str(num)][1].tricontour(target_2d_profiled_data[str(num)]['all_WCs'][:, names[WC2d_2[num]]], target_2d_profiled_data[str(num)]['all_WCs'][:, names[WC2d_1[num]]], target_2d_profiled_data[str(num)]['2dNLL'], colors='k', linestyles=['dashed', 'dashdot', 'dotted'], levels=[2.30, 6.18, 11.83]) # 1, 2, and 3 sigmas
    model_contour = fake_profiled_2d_graphs[str(num)][1].tricontour(model_2d_fake_profiled_data[str(num)]['all_WCs'][:, names[WC2d_2[num]]], model_2d_fake_profiled_data[str(num)]['all_WCs'][:, names[WC2d_1[num]]], model_2d_fake_profiled_data[str(num)]['2dNLL'], colors='r', linestyles=['dashed', 'dashdot', 'dotted'], levels=[2.30, 6.18, 11.83]) # 1, 2, and 3 sigmas
    SM_value = fake_profiled_2d_graphs[str(num)][1].scatter(0, 0, marker='d', c='gold', ec='royalblue', s=30, linewidths=1, zorder=10)
    fake_profiled_2d_graphs[str(num)][1].legend(target_contour.collections+model_contour.collections+[SM_value], ['$1\sigma$ target', '$2\sigma$ target', '$3\sigma$ target', '$1\sigma$ predicted', '$2\sigma$ predicted', '$3\sigma$ predicted', 'SM value'])
    fake_profiled_2d_graphs[str(num)][1].set_xlabel(WC2d_2[num])
    fake_profiled_2d_graphs[str(num)][1].set_ylabel(WC2d_1[num])
    fake_profiled_2d_graphs[str(num)][1].set_title('Fake Profiled')
    fake_profiled_2d_graphs[str(num)][0].tight_layout()

# Save the graphs
for key in fake_profiled_2d_graphs.keys():
    pp.savefig(fake_profiled_2d_graphs[key][0])

# 1D real profiled graphs compared to Combine scans
profiled_1d_graphs = {}
for key in names.keys():
    # Now have to shift because training data is shifted to 0
    target_1d_profiled_data[key]['2dNLL'] -= target_1d_profiled_data[key]['2dNLL'].min()
    model_1d_profiled_data[key]['2dNLL'] -= model_1d_profiled_data[key]['2dNLL'].min()
    profiled_1d_graphs[key] = plt.subplots()
    target_scatter = profiled_1d_graphs[key][1].scatter(target_1d_profiled_data[key]['all_WCs'][:,names[key]], target_1d_profiled_data[key]['2dNLL'], marker='^', c='none', ec='k', s=5, linewidths=0.2)
    model_scatter = profiled_1d_graphs[key][1].scatter(model_1d_profiled_data[key]['all_WCs'][:,names[key]], model_1d_profiled_data[key]['2dNLL'], marker='v', c='none', ec='g', s=5, linewidths=0.2)
    one_line, = profiled_1d_graphs[key][1].plot([target_1d_profiled_data[key]['all_WCs'][:,names[key]].min(), target_1d_profiled_data[key]['all_WCs'][:,names[key]].max()], [1,1], 'r--', linewidth=0.5)
    four_line, = profiled_1d_graphs[key][1].plot([target_1d_profiled_data[key]['all_WCs'][:,names[key]].min(), target_1d_profiled_data[key]['all_WCs'][:,names[key]].max()], [4,4], 'm:', linewidth=0.5)
    profiled_1d_graphs[key][1].legend([target_scatter, model_scatter, one_line, four_line], ['Target', 'NN Prediction', '$1\sigma\ (2\Delta NLL=1)$', '$2\sigma\ (2\Delta NLL=4)$'], markerscale=3)
    profiled_1d_graphs[key][1].set_xlabel(key)
    profiled_1d_graphs[key][1].set_ylabel('2$\Delta$NLL')
    profiled_1d_graphs[key][1].set_title('Profiled')
    profiled_1d_graphs[key][0].tight_layout()

zoomed_profiled_1d_graphs = {}
for key in names.keys():
    target_1d_zoomed_profiled_data[key]['2dNLL'] -= target_1d_zoomed_profiled_data[key]['2dNLL'].min()
    model_1d_zoomed_profiled_data[key]['2dNLL'] -= model_1d_zoomed_profiled_data[key]['2dNLL'].min()
    zoomed_profiled_1d_graphs[key] = plt.subplots()
    target_scatter = zoomed_profiled_1d_graphs[key][1].scatter(target_1d_zoomed_profiled_data[key]['all_WCs'][:,names[key]], target_1d_zoomed_profiled_data[key]['2dNLL'], marker='^', c='none', ec='k', s=5, linewidths=0.2)
    model_scatter = zoomed_profiled_1d_graphs[key][1].scatter(model_1d_zoomed_profiled_data[key]['all_WCs'][:,names[key]], model_1d_zoomed_profiled_data[key]['2dNLL'], marker='v', c='none', ec='g', s=5, linewidths=0.2)
    one_line, = zoomed_profiled_1d_graphs[key][1].plot([target_1d_zoomed_profiled_data[key]['all_WCs'][:,names[key]].min(), target_1d_zoomed_profiled_data[key]['all_WCs'][:,names[key]].max()], [1,1], 'r--', linewidth=0.5)
    four_line, = zoomed_profiled_1d_graphs[key][1].plot([target_1d_zoomed_profiled_data[key]['all_WCs'][:,names[key]].min(), target_1d_zoomed_profiled_data[key]['all_WCs'][:,names[key]].max()], [4,4], 'm:', linewidth=0.5)
    zoomed_profiled_1d_graphs[key][1].legend([target_scatter, model_scatter, one_line, four_line], ['Target', 'NN Prediction', '$1\sigma\ (2\Delta NLL=1)$', '$2\sigma\ (2\Delta NLL=4)$'], markerscale=3)
    zoomed_profiled_1d_graphs[key][1].set_xlabel(key)
    zoomed_profiled_1d_graphs[key][1].set_ylabel('2$\Delta$NLL')
    zoomed_profiled_1d_graphs[key][1].set_title('Profiled')
    zoomed_profiled_1d_graphs[key][0].tight_layout()

# Save the graphs
for key in profiled_1d_graphs.keys():
    pp.savefig(profiled_1d_graphs[key][0])
for key in zoomed_profiled_1d_graphs.keys():
    pp.savefig(zoomed_profiled_1d_graphs[key][0])

# 2D real profiled graphs compared to Combine scans
profiled_2d_graphs = {}
for num in nums:
    target_2d_profiled_data[str(num)]['2dNLL'] -= target_2d_profiled_data[str(num)]['2dNLL'].min()
    model_2d_profiled_data[str(num)]['2dNLL'] -= model_2d_profiled_data[str(num)]['2dNLL'].min()
    profiled_2d_graphs[str(num)] = plt.subplots()
    target_contour = profiled_2d_graphs[str(num)][1].tricontour(target_2d_profiled_data[str(num)]['all_WCs'][:, names[WC2d_2[num]]], target_2d_profiled_data[str(num)]['all_WCs'][:, names[WC2d_1[num]]], target_2d_profiled_data[str(num)]['2dNLL'], colors='k', linestyles=['dashed', 'dashdot', 'dotted'], levels=[2.30, 6.18, 11.83]) # 1, 2, and 3 sigmas
    model_contour = profiled_2d_graphs[str(num)][1].tricontour(model_2d_profiled_data[str(num)]['all_WCs'][:, names[WC2d_2[num]]], model_2d_profiled_data[str(num)]['all_WCs'][:, names[WC2d_1[num]]], model_2d_profiled_data[str(num)]['2dNLL'], colors='r', linestyles=['dashed', 'dashdot', 'dotted'], levels=[2.30, 6.18, 11.83]) # 1, 2, and 3 sigmas
    SM_value = profiled_2d_graphs[str(num)][1].scatter(0, 0, marker='d', c='gold', ec='royalblue', s=30, linewidths=1, zorder=10)
    profiled_2d_graphs[str(num)][1].legend(target_contour.collections+model_contour.collections+[SM_value], ['$1\sigma$ target', '$2\sigma$ target', '$3\sigma$ target', '$1\sigma$ predicted', '$2\sigma$ predicted', '$3\sigma$ predicted', 'SM value'])
    profiled_2d_graphs[str(num)][1].set_xlabel(WC2d_2[num])
    profiled_2d_graphs[str(num)][1].set_ylabel(WC2d_1[num])
    profiled_2d_graphs[str(num)][1].set_title('Profiled')
    profiled_2d_graphs[str(num)][0].tight_layout()

# Save the graphs
for key in profiled_2d_graphs.keys():
    pp.savefig(profiled_2d_graphs[key][0])

# Save the graphs and data to files
np.savez(f'./graphs/{out_file}_validation.npz', **data_dict)
pp.close()