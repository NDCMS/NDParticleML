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
from matplotlib.backends.backend_pdf import PdfPages

torch.set_printoptions(threshold=10000) # So big tensors can be printed without summary

# Functions
def evaluate(inputs):
    # Assumes model uses square and cross terms as well as standardization
    # Currently only uses a slice of inputs_all
    all_shape = list(inputs.shape) # The shape of the final input tensor, including the square and cross terms
    all_shape[-1] = 152
    cross_shape = list(inputs.shape) # The shape of the input cross terms tensor
    cross_shape[-1] = 120
    inputs_all = torch.zeros(all_shape).cuda()
    squares = inputs**2
    cross_terms = torch.zeros(cross_shape).cuda()
    idx = 0
    for i in range(16):
        for j in range(i):
            cross_terms[...,idx] = inputs[...,i] * inputs[...,j]
            idx += 1
    inputs_all[...,0:16] = inputs
    inputs_all[...,16:32] = squares
    inputs_all[...,32:152] = cross_terms
    std_inputs = nnm.affine_transform(inputs_all, input_stats)
    std_outputs = model(std_inputs)
    outputs = nnm.affine_untransform(std_outputs, output_stats)
    
    return outputs

out_file = '3169_0'

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

save_dict = torch.load(f'./{out_file}_model+.pt')
best_model_state = save_dict['model']
parameters_save = save_dict['parameters']
input_stats = save_dict['input_stats']
output_stats = save_dict['output_stats']

model = nnm.create_model(152, 1, parameters_save) # Hard-coded for the inclusion of square and cross terms
model.load_state_dict(best_model_state)
model.eval()

epochs = 100
random_starting_points = 100 # The number of random starting points to do gradient descent on for each scanned value

actual_profiled_graph_data = {}
for key in name_list:
    loaded = np.load(f'likelihood_profiled_{key}.npz') # A dictionary with WC names as keys
    inputs = np.zeros((loaded['deltaNLL'].shape[0], random_starting_points, 16))
    for key2 in names.keys():
        inputs[...,names[key2]] = loaded[key2][:,np.newaxis]
    actual_profiled_graph_data[key] = {key: inputs, 'deltaNLL': loaded['deltaNLL']}
    # actual_profiled_graph_data[key]['deltaNLL'] -= actual_profiled_graph_data[key]['deltaNLL'].min()
    actual_profiled_graph_data[key]['deltaNLL'] *= 2

actual_zoomed_profiled_graph_data = {}
for key in name_list:
    less_than_10 = (actual_profiled_graph_data[key]['deltaNLL'] < 10)
    actual_zoomed_profiled_graph_data[key] = {key: actual_profiled_graph_data[key][key][less_than_10], 'deltaNLL': actual_profiled_graph_data[key]['deltaNLL'][less_than_10]}

model_zoomed_profiled_graph_data = {}

for key in name_list:
    torch.autograd.set_detect_anomaly(True)
    inputs_old = actual_zoomed_profiled_graph_data[key][key]
    model_zoomed_profiled_graph_data[key] = {key: inputs_old[..., 0, names[key]], 'deltaNLL': np.zeros_like(inputs_old[..., 0, names[key]])}
    num_inputs = inputs_old.shape[0]
    inputs = (np.random.random_sample(inputs_old.shape) - 0.5) * 40
    inputs[...,names[key]] = inputs_old[...,names[key]] # copy over the WC being scanned, while leaving the other 15 randomized
    inputs = torch.from_numpy(inputs).cuda()
    inputs.requires_grad = True

    min_outputs = evaluate(inputs) # The outputs of the random starting points, to be updated every epoch
    min_WCs = inputs.detach().clone() # A snapshot of the WCs of all the points to scan

    optimizer = torch.optim.Adam([inputs],lr=2e-0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, threshold=1e-6)
    optimizer.zero_grad()
    for epoch in range(epochs):
        outputs = evaluate(inputs)
        outputs_cp = outputs.detach().clone()
        outputs_sum = torch.sum(outputs)
        idx_to_update = torch.where(outputs_cp < min_outputs)[0]
        min_outputs[idx_to_update] = outputs_cp[idx_to_update]
        min_WCs[idx_to_update] = inputs.detach().clone()[idx_to_update]
        optimizer.zero_grad()
        outputs_sum.backward()
        inputs.grad[...,names[key]] = 0
        optimizer.step()
        #scheduler.step(output)
    (min_outputs_scanned, min_starting_point_indicies) = torch.min(min_outputs, -2) # Get the best all starting points

    # Make the index tensor suitable for gathering the min_WCs
    min_starting_point_indicies = min_starting_point_indicies.unsqueeze(-2)
    min_starting_point_indicies_shape = list(min_starting_point_indicies.shape)
    min_starting_point_indicies_shape[-1] = 16
    min_starting_point_indicies = min_starting_point_indicies.expand(min_starting_point_indicies_shape) # Warning: don't perform in-place operations on this since expand() does not allocate new memory

    min_WCs_scanned = torch.gather(min_WCs, -2, min_starting_point_indicies) # Get the WCs corresponding to the best-performing starting points
    model_zoomed_profiled_graph_data[key]['deltaNLL'] = min_outputs_scanned.cpu().detach().numpy().flatten() * 2
    print (key)
    print (min_WCs_scanned)

zoomed_profiled_graphs = {}
for key in name_list:
    zoomed_profiled_graphs[key] = plt.subplots()
    actual_scatter = zoomed_profiled_graphs[key][1].scatter(model_zoomed_profiled_graph_data[key][key], actual_zoomed_profiled_graph_data[key]['deltaNLL'], marker='^', c='none', ec='k', s=5, linewidths=0.2)
    model_scatter = zoomed_profiled_graphs[key][1].scatter(model_zoomed_profiled_graph_data[key][key], model_zoomed_profiled_graph_data[key]['deltaNLL'], marker='v', c='none', ec='g', s=5, linewidths=0.2)
    one_line, = zoomed_profiled_graphs[key][1].plot([model_zoomed_profiled_graph_data[key][key].min(), model_zoomed_profiled_graph_data[key][key].max()], [1,1], 'r-', linewidth=0.5)
    four_line, = zoomed_profiled_graphs[key][1].plot([model_zoomed_profiled_graph_data[key][key].min(), model_zoomed_profiled_graph_data[key][key].max()], [4,4], 'm:', linewidth=0.5)
    zoomed_profiled_graphs[key][1].legend([actual_scatter, model_scatter, one_line, four_line], ['Target', 'NN Prediction', '$1\sigma\ (2\Delta NLL=1)$', '$2\sigma\ (2\Delta NLL=4)$'], markerscale=3)
    zoomed_profiled_graphs[key][1].set_xlabel(key)
    zoomed_profiled_graphs[key][1].set_ylabel('2$\Delta$NLL')
    zoomed_profiled_graphs[key][1].set_title('Profiled')
    zoomed_profiled_graphs[key][0].tight_layout()

pp = PdfPages(f'./graphs/{out_file}_validation.pdf')
for key in name_list:
    pp.savefig(zoomed_profiled_graphs[key][0])
pp.close()

# Bug to fix in WC output
# Change DeltaNLL zeros-like to something positive