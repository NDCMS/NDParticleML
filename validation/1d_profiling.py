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

# Functions
def evaluate(inputs, model, input_stats, output_stats):
    # Assumes model was trained using square and cross terms as well as standardization
    # Assumes model takes in 16 terms and an internal layer calculates the square and cross terms
    # Currently only uses a slice of inputs_all
    std_inputs = nnm.affine_transform(inputs, input_stats)
    std_outputs = model(std_inputs)
    outputs = nnm.affine_untransform(std_outputs, output_stats)

    return outputs

out_file = '12643_0'

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

name_list = list(names)

batch_size = 4096
epochs = 100
random_starting_points = 100 # The number of random starting points to do gradient descent on for each scanned value

save_dict = torch.load(f'./{out_file}_model+.pt')
best_model_state = save_dict['model']
parameters_save = save_dict['parameters']
input_stats = save_dict['input_stats']
output_stats = save_dict['output_stats']

model = nnm.create_model(parameters_save['N'], 1, parameters_save)
model.load_state_dict(best_model_state)
model.eval()

# Only make selective graphs
nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# actual frozen data
actual_frozen_data = {}
for num in nums:
    WC = name_list[num]
    loaded = np.load(f'likelihood_{WC}.npz')
    actual_frozen_data[str(num)] = {WC: loaded[WC], 'deltaNLL': loaded['deltaNLL']}
    # Shift minimum to 0 for frozen graphs
    actual_frozen_data[str(num)]['deltaNLL'] -= actual_frozen_data[str(num)]['deltaNLL'].min()
    actual_frozen_data[str(num)]['deltaNLL'] *= 2
    
# actual profiled data
actual_profiled_data = {}
for num in nums:
    WC = name_list[num]
    loaded = np.load(f'likelihood_profiled_{WC}.npz')
    inputs = np.zeros((loaded['deltaNLL'].shape[0], random_starting_points, 16))
    for key2 in name_list: # Need all 16 even if only care about one graph
        inputs[...,names[key2]] = loaded[key2][:,np.newaxis] # broadcast into the new axis of random starting points
    actual_profiled_data[str(num)] = {'inputs': inputs, 'deltaNLL': loaded['deltaNLL']} # So inputs contains all inputs, not just for graphing
    # Don't shift for profiled graphs!
    #actual_profiled_data[str(num)]['deltaNLL'] -= actual_profiled_data[str(num)]['deltaNLL'].min()
    actual_profiled_data[str(num)]['deltaNLL'] *= 2

# actual zoomed frozen data
actual_zoomed_frozen_data = {}
for num in nums:
    WC = name_list[num]
    less_than_10 = (actual_frozen_data[str(num)]['deltaNLL'] < 10)
    actual_zoomed_frozen_data[str(num)] = {WC: actual_frozen_data[str(num)][WC][less_than_10], 'deltaNLL': actual_frozen_data[str(num)]['deltaNLL'][less_than_10]}

# actual zoomed profiled data
actual_zoomed_profiled_data = {}
for num in nums:
    WC = name_list[num]
    less_than_10 = (actual_profiled_data[str(num)]['deltaNLL'] < 10)
    actual_zoomed_profiled_data[str(num)] = {'inputs': actual_profiled_data[str(num)]['inputs'][less_than_10], 'deltaNLL': actual_profiled_data[str(num)]['deltaNLL'][less_than_10]}
    
# model frozen data
model_frozen_data = {}
for num in nums:
    WC = name_list[num]
    inputs = actual_frozen_data[str(num)][WC]
    num_inputs = inputs.shape[0]
    inputs_all = np.zeros((num_inputs, 16))
    inputs_all[:,names[WC]] = inputs

    std_inputs = nnm.affine_transform(torch.from_numpy(inputs_all).float().cuda(), input_stats)
    std_outputs = torch.full((num_inputs, 1), 100.).cuda() # fill outputs with 100. as a default (decimal to force dtype=float not int)
    inputMiniBatches = torch.split(std_inputs, batch_size)
    batch_idx = 0
    for minibatch in range(len(inputMiniBatches)):
        batch_outputs = model(inputMiniBatches[minibatch])
        std_outputs[batch_idx: batch_idx + batch_outputs.shape[0]] = batch_outputs
        batch_idx += batch_outputs.shape[0]
    outputs = nnm.affine_untransform(std_outputs, output_stats).cpu().detach().numpy().flatten()
    outputs -= outputs.min()
    outputs *= 2
    model_frozen_data[str(num)] = {WC: inputs, 'deltaNLL': outputs}
    
# model profiled data
model_profiled_data = {}
for num in nums:
    WC = name_list[num]
    torch.autograd.set_detect_anomaly(True)
    inputs_old = actual_profiled_data[str(num)]['inputs']
    num_inputs = inputs_old.shape[0]
    model_profiled_data[str(num)] = {WC: inputs_old[..., 0, num]}
    inputs = (np.random.random_sample(inputs_old.shape) - 0.5) * 40
    inputs[...,num] = inputs_old[...,num] # copy over the WC being scanned, while leaving the other 15 randomized
    inputs = torch.from_numpy(inputs).float().cuda()
    inputs.requires_grad = True

    outputs = torch.full((num_inputs, 1), 100.).cuda() # fill outputs with 100. as a default (decimal to force dtype=float not int)
    inputMiniBatches = torch.split(inputs, batch_size)
    batch_idx = 0

    start_time = time.perf_counter()

    # Memory debugging info
    memory_fig, memory_ax = plt.subplots()
    memory_count = 0
    memory_count_arr = np.full((len(inputMiniBatches)*epochs), np.NaN)
    memory_vals = np.full((len(inputMiniBatches)*epochs), np.NaN)

    for minibatch in range(len(inputMiniBatches)):
        optimizer = torch.optim.Adam([inputs],lr=2e-0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, threshold=1e-6)
        inputMiniBatches = torch.split(inputs, batch_size)
        batch_inputs = inputMiniBatches[minibatch]
        min_outputs = evaluate(batch_inputs, model, input_stats, output_stats) # The outputs of the random starting points, to be updated every epoch
        min_WCs = batch_inputs.detach().clone() # A snapshot of the WCs of all the points to scan
        optimizer.zero_grad()
        print (f'Starting {minibatch}/{len(inputMiniBatches)} minibatches.')
        for epoch in range(epochs):
            #print (f'Starting {epoch}/{epochs} epochs of {minibatch}/{len(inputMiniBatches)} minibatches.')

            inputMiniBatches = torch.split(inputs, batch_size)
            batch_inputs = inputMiniBatches[minibatch]
            batch_outputs = evaluate(batch_inputs, model, input_stats, output_stats)
            batch_outputs_cp = batch_outputs.detach().clone()
            batch_outputs_sum = torch.sum(torch.log(batch_outputs + 10)) # Optimize the sum of outputs
            idx_to_update = torch.where(batch_outputs_cp < min_outputs)[0]
            min_outputs[idx_to_update] = batch_outputs_cp[idx_to_update]
            min_WCs[idx_to_update] = batch_inputs.detach().clone()[idx_to_update]
            optimizer.zero_grad()
            batch_outputs_sum.backward()
            #print (torch.count_nonzero(inputs.grad))
            #print (inputs.grad)
            #print (inputs.grad[1500,31,6])
            #print (f'Sum of minibatch outputs: {batch_outputs_sum}')
            #print (f'Output of a random point: {batch_outputs[500,0,0]}')
            inputs.grad[...,num] = 0
            optimizer.step()
            #scheduler.step(output)

            # Memory debugging info
            memory_count_arr[memory_count] = memory_count
            memory_vals[memory_count] = torch.cuda.memory_allocated()/1e9
            memory_ax.cla()
            memory_ax.plot(memory_count_arr, memory_vals)
            memory_count += 1

        (min_outputs_scanned, min_starting_point_indicies) = torch.min(min_outputs, -2) # Get the best all starting points

        # Make the index tensor suitable for gathering the min_WCs
        min_starting_point_indicies = min_starting_point_indicies.unsqueeze(-2)
        min_starting_point_indicies_shape = list(min_starting_point_indicies.shape)
        min_starting_point_indicies_shape[-1] = 16
        min_starting_point_indicies = min_starting_point_indicies.expand(min_starting_point_indicies_shape) # Warning: don't perform in-place operations on this since expand() does not allocate new memory

        min_WCs_scanned = torch.gather(min_WCs, -2, min_starting_point_indicies) # Get the WCs corresponding to the best-performing starting points
        outputs[batch_idx: batch_idx + batch_outputs.shape[0]] = min_outputs_scanned.detach().clone() # detach from graph to delete obsolete graphs from memory! This was the culprit causing the memory leak
        print (WC + ' ' + 'batch number: ' + str(minibatch))
        print (min_WCs_scanned)
        batch_idx += batch_outputs.shape[0]
    print ('Profiling done!')
    print (f'Time used: {time.perf_counter() - start_time} seconds.')
    outputs = outputs.cpu().detach().numpy().flatten()
    # Don't shift minimum to 0 for 1D profiled scans
    #outputs -= outputs.min()
    outputs *= 2
    model_profiled_data[str(num)]['deltaNLL'] = outputs

# model zoomed frozen data
model_zoomed_frozen_data = {}
for num in nums:
    WC = name_list[num]
    less_than_10 = (actual_frozen_data[str(num)]['deltaNLL'] < 10)
    model_zoomed_frozen_data[str(num)] = {WC: model_frozen_data[str(num)][WC][less_than_10], 'deltaNLL': model_frozen_data[str(num)]['deltaNLL'][less_than_10]}

# model zoomed profiled data
model_zoomed_profiled_data = {}
for num in nums:
    WC = name_list[num]
    less_than_10 = (actual_profiled_data[str(num)]['deltaNLL'] < 10)
    model_zoomed_profiled_data[str(num)] = {WC: model_profiled_data[str(num)][WC][less_than_10], 'deltaNLL': model_profiled_data[str(num)]['deltaNLL'][less_than_10]}

# diff data
frozen_diff_data = copy.deepcopy(actual_frozen_data)
profiled_diff_data = copy.deepcopy(model_profiled_data) # since actual_profiled_2D_data actually has all 16 variables stored in it
for num in nums:
    frozen_diff_data[str(num)]['deltaNLL'] = model_frozen_data[str(num)]['deltaNLL'] - actual_frozen_data[str(num)]['deltaNLL']
    profiled_diff_data[str(num)]['deltaNLL'] = model_profiled_data[str(num)]['deltaNLL'] - actual_profiled_data[str(num)]['deltaNLL']
    
# Make graphs

frozen_graphs = {}
profiled_graphs = {}
zoomed_frozen_graphs = {}
zoomed_profiled_graphs = {}
frozen_diff_graphs = {}
profiled_diff_graphs = {}
for num in nums:
    WC = name_list[num]
    frozen_graphs[WC] = plt.subplots()
    actual_scatter = frozen_graphs[WC][1].scatter(actual_frozen_data[str(num)][WC], actual_frozen_data[str(num)]['deltaNLL'], marker='.', c='b', s=0.5, linewidths=0.5)
    model_scatter = frozen_graphs[WC][1].scatter(model_frozen_data[str(num)][WC], model_frozen_data[str(num)]['deltaNLL'], marker='x', c='g', s=1, linewidths=0.2)
    one_line, = frozen_graphs[WC][1].plot([model_frozen_data[str(num)][WC].min(), model_frozen_data[str(num)][WC].max()], [1,1], 'r--', linewidth=0.5)
    four_line, = frozen_graphs[WC][1].plot([model_frozen_data[str(num)][WC].min(), model_frozen_data[str(num)][WC].max()], [4,4], 'm:', linewidth=0.5)
    frozen_graphs[WC][1].legend([actual_scatter, model_scatter, one_line, four_line], ['Target', 'NN Prediction', '$1\sigma\ (2\Delta NLL=1)$', '$2\sigma\ (2\Delta NLL=4)$'], markerscale=6)
    frozen_graphs[WC][1].set_xlabel(WC)
    frozen_graphs[WC][1].set_ylabel('2$\Delta$NLL')
    frozen_graphs[WC][1].set_title('Frozen')
    frozen_graphs[WC][0].tight_layout()

for num in nums:
    WC = name_list[num]
    profiled_graphs[WC] = plt.subplots()
    actual_scatter = profiled_graphs[WC][1].scatter(model_profiled_data[str(num)][WC], actual_profiled_data[str(num)]['deltaNLL'], marker='.', c='b', s=0.5, linewidths=0.5)
    model_scatter = profiled_graphs[WC][1].scatter(model_profiled_data[str(num)][WC], model_profiled_data[str(num)]['deltaNLL'], marker='x', c='g', s=1, linewidths=0.2)
    one_line, = profiled_graphs[WC][1].plot([model_profiled_data[str(num)][WC].min(), model_profiled_data[str(num)][WC].max()], [1,1], 'r--', linewidth=0.5)
    four_line, = profiled_graphs[WC][1].plot([model_profiled_data[str(num)][WC].min(), model_profiled_data[str(num)][WC].max()], [4,4], 'm:', linewidth=0.5)
    profiled_graphs[WC][1].legend([actual_scatter, model_scatter, one_line, four_line], ['Target', 'NN Prediction', '$1\sigma\ (2\Delta NLL=1)$', '$2\sigma\ (2\Delta NLL=4)$'], markerscale=6)
    profiled_graphs[WC][1].set_xlabel(WC)
    profiled_graphs[WC][1].set_ylabel('2$\Delta$NLL')
    profiled_graphs[WC][1].set_title('Profiled')
    profiled_graphs[WC][0].tight_layout()
    
for num in nums:
    WC = name_list[num]
    zoomed_frozen_graphs[WC] = plt.subplots()
    actual_scatter = zoomed_frozen_graphs[WC][1].scatter(actual_zoomed_frozen_data[str(num)][WC], actual_zoomed_frozen_data[str(num)]['deltaNLL'], marker='.', c='b', s=0.5, linewidths=0.5)
    model_scatter = zoomed_frozen_graphs[WC][1].scatter(model_zoomed_frozen_data[str(num)][WC], model_zoomed_frozen_data[str(num)]['deltaNLL'], marker='x', c='g', s=1, linewidths=0.2)
    one_line, = zoomed_frozen_graphs[WC][1].plot([model_zoomed_frozen_data[str(num)][WC].min(), model_zoomed_frozen_data[str(num)][WC].max()], [1,1], 'r--', linewidth=0.5)
    four_line, = zoomed_frozen_graphs[WC][1].plot([model_zoomed_frozen_data[str(num)][WC].min(), model_zoomed_frozen_data[str(num)][WC].max()], [4,4], 'm:', linewidth=0.5)
    zoomed_frozen_graphs[WC][1].legend([actual_scatter, model_scatter, one_line, four_line], ['Target', 'NN Prediction', '$1\sigma\ (2\Delta NLL=1)$', '$2\sigma\ (2\Delta NLL=4)$'], markerscale=6)
    zoomed_frozen_graphs[WC][1].set_xlabel(WC)
    zoomed_frozen_graphs[WC][1].set_ylabel('2$\Delta$NLL')
    zoomed_frozen_graphs[WC][1].set_title('Frozen')
    zoomed_frozen_graphs[WC][0].tight_layout()
    
for num in nums:
    WC = name_list[num]
    zoomed_profiled_graphs[WC] = plt.subplots()
    actual_scatter = zoomed_profiled_graphs[WC][1].scatter(model_zoomed_profiled_data[str(num)][WC], actual_zoomed_profiled_data[str(num)]['deltaNLL'], marker='.', c='b', s=0.5, linewidths=0.5)
    model_scatter = zoomed_profiled_graphs[WC][1].scatter(model_zoomed_profiled_data[str(num)][WC], model_zoomed_profiled_data[str(num)]['deltaNLL'], marker='x', c='g', s=1, linewidths=0.2)
    one_line, = zoomed_profiled_graphs[WC][1].plot([model_zoomed_profiled_data[str(num)][WC].min(), model_zoomed_profiled_data[str(num)][WC].max()], [1,1], 'r--', linewidth=0.5)
    four_line, = zoomed_profiled_graphs[WC][1].plot([model_zoomed_profiled_data[str(num)][WC].min(), model_zoomed_profiled_data[str(num)][WC].max()], [4,4], 'm:', linewidth=0.5)
    zoomed_profiled_graphs[WC][1].legend([actual_scatter, model_scatter, one_line, four_line], ['Target', 'NN Prediction', '$1\sigma\ (2\Delta NLL=1)$', '$2\sigma\ (2\Delta NLL=4)$'], markerscale=6)
    zoomed_profiled_graphs[WC][1].set_xlabel(WC)
    zoomed_profiled_graphs[WC][1].set_ylabel('2$\Delta$NLL')
    zoomed_profiled_graphs[WC][1].set_title('Profiled')
    zoomed_profiled_graphs[WC][0].tight_layout()

for num in nums:
    WC = name_list[num]
    frozen_diff_graphs[WC] = plt.subplots()
    frozen_diff_graphs[WC][1].scatter(frozen_diff_data[str(num)][WC], frozen_diff_data[str(num)]['deltaNLL'], c='b', s=1)
    frozen_diff_graphs[WC][1].set_xlabel(WC)
    frozen_diff_graphs[WC][1].set_ylabel('prediction - target')
    frozen_diff_graphs[WC][1].set_title('Frozen')
    frozen_diff_graphs[WC][0].tight_layout()

for num in nums:
    WC = name_list[num]
    profiled_diff_graphs[WC] = plt.subplots()
    profiled_diff_graphs[WC][1].scatter(profiled_diff_data[str(num)][WC], profiled_diff_data[str(num)]['deltaNLL'], c='b', s=1)
    profiled_diff_graphs[WC][1].set_xlabel(WC)
    profiled_diff_graphs[WC][1].set_ylabel('prediction - target')
    profiled_diff_graphs[WC][1].set_title('Profiled')
    profiled_diff_graphs[WC][0].tight_layout()

# Save graphs to pdf
pp = PdfPages(f'./graphs/{out_file}_validation_with_profiling.pdf')
for key in frozen_graphs.keys():
    pp.savefig(frozen_graphs[key][0])
for key in profiled_graphs.keys():
    pp.savefig(profiled_graphs[key][0])
for key in zoomed_frozen_graphs.keys():
    pp.savefig(zoomed_frozen_graphs[key][0])
for key in zoomed_profiled_graphs.keys():
    pp.savefig(zoomed_profiled_graphs[key][0])
for key in frozen_diff_graphs.keys():
    pp.savefig(frozen_diff_graphs[key][0])
for key in profiled_diff_graphs.keys():
    pp.savefig(profiled_diff_graphs[key][0])
pp.close()