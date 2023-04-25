# %matplotlib notebook
import torch
from torch import nn
import numpy as np
import numpy.ma as ma
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import time
import copy
import pandas as pd
from pandas import read_csv
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors as colors
import argparse
import polynomial as poly

# Functions

# Sets error calculations
def abs_err(pred, act):
    """
    Returns the absolute error of two numbers.
    Inputs: pred (float), act (float)
    Outputs: the absolute error (float)
    """
    return abs(pred-act)
def rel_err(pred, act):
    """
    Returns the relative error of two numbers.
    Inputs: pred (float), act (float)
    Outputs: the relative error (float)
    """
    if act == 0: return 9e+10
    return abs(abs_err(pred, act)/act)

# Deprecated. Use LinearLayer instead.
def affine_transform(tensor, stats):
    """
    Subtracts mean and divides by standard deviation.
    Inputs: tensor (Pytorch tensor), stats (mean, standard deviation) (tuple)
    Outputs: tensor (Pytorch tensor)
    """
    return (tensor - stats[0]) / stats[1]

# Deprecated. Use LinearLayer instead.
def affine_untransform(tensor, stats):
    """
    Inverse function of affine_transform. Multiplies by stddev and adds mean.
    Inputs: tensor (Pytorch tensor), stats (mean, standard deviation) (tuple)
    Outputs: tensor (Pytorch tensor)
    """
    return tensor * stats[1] + stats[0]

def find_stats(tensor):
    """
    Finds the mean and standard deviation per coordinate.
    Inputs: tensor (Pytorch tensor)
    Outputs: (mean, stddev) (tuple)
    """
    mean = torch.mean(tensor, 0)
    stddev = torch.std(tensor, 0)

    return (mean, stddev)

# Check if a prediction is within 0.01 absolute accuracy or 1% relative accuracy
def accu_test(prediction, actual):
    """
    Tests if two numbers are close enough (0.01 absolute error or 1% relative error).
    Inputs: prediction (float), actual (float)
    Outputs: 1 if close enough, 0 if not close enough (integer)
    """
    if (abs_err(prediction, actual) < 0.05 or rel_err(prediction, actual) < 0.01):
        return 1
    else: return 0
v_accu_test = np.vectorize(accu_test) # This makes a vector function for convenience

# Create a simple neural network with layer and node variabliltiy
def create_model(input_dim, output_dim, parameters, input_stats, output_stats, lin_trans = None):
    """
    Creates a sequential model with the same number of nodes in each hidden layer. Adds a linear transformation of inputs in the front and sandwiches the other layers between standardization layers.

    linear (fixed) -> standardization (fixed) -> hidden layers -> unstandardization

    Inputs: input_dim (integer, the number of variables), output_dim (integer), parameters (dictionary), input_stats (mean, standard deviation) (tuple), output_stats (mean, standard deviation) (tuple), lin_trans (tensor)
    Outputs: model (Pytorch sequential container)
    """
    layers = []
    if (lin_trans is None):
        # Add an identity matrix layer so that the saved model will have the correct layer 
        # numbers even if we add a nontrivial linear layer later
        layers.append(LinearLayer(torch.eye(input_dim), torch.zeros(input_dim).cuda()))
    else:
        layers.append(LinearLayer(lin_trans, torch.zeros(input_dim).cuda()))
    layers.append(LinearLayer(torch.diag(1/input_stats[1]), -input_stats[0]/input_stats[1]))
    #Checks if this model will have polynomial layer. If so, then it makes one based on the given degrees.
    if (parameters['polynomial']):
        layers += [poly.PolynomialLayer(input_dim,parameters['polynomial_degree'],parameters['hidden_nodes']), torch.nn.ReLU()]
    else:
        layers += [torch.nn.Linear(input_dim,parameters['hidden_nodes']),torch.nn.ReLU()]
    for i in range(parameters['hidden_layers']):
        layers.append(torch.nn.Linear(parameters['hidden_nodes'],parameters['hidden_nodes']))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(parameters['hidden_nodes'],output_dim)) # We only care about functions with one output
    layers.append(LinearLayer(torch.diag(output_stats[1]), output_stats[0]))
    model = torch.nn.Sequential(*layers)
    # include different number of nodes per layer functionality
    #list with nodes per layer
    return model.cuda()

# Train network
def train_network(model, inputs_train, outputs_train, inputs_test, outputs_test, parameters, show_progress = True):
    """
    Trains a network of a given architecture.
    Inputs: model (Pytorch sequential container), inputs_train (training input data; Pytorch tensor), outputs_train (training output data; Pytorch tensor), inputs_test (testing input data; Pytorch tensor), outputs_test (testing output data; Pytorch tensor), parameters (dictionary), show_progress (boolean)
    Outputs: (graph_data, best_model_state, parameters_save): tuple (dictionary, state dictionary, dictionary)
    """
    # Useful information
    total_num = torch.numel(outputs_test)
    N = inputs_train.shape[1]
    outputs_test_np = outputs_test.cpu().detach().numpy().flatten()
    test_size = outputs_test_np.size

    # Create a representative training set for evaluation
    index_std_train_rep = np.random.choice(parameters['train_size'], parameters['test_size'], replace=False)
    inputs_train_rep = inputs_train[index_std_train_rep]
    outputs_train_rep = outputs_train[index_std_train_rep]

    # Useful info for accu_out graph
    # Calculate accuracy for each region
    max_out = outputs_test_np.max()
    min_out = outputs_test_np.min()
    max_graph = max_out + (max_out - min_out) / 100
    min_graph = min_out - (max_out - min_out) / 100
    grid_size = (max_graph - min_graph) / parameters['accu_out_resolution']
    grid_accu_tally = np.zeros((parameters['accu_out_resolution'], parameters['n_epochs'], 2))
    grid_num = np.floor((outputs_test_np - min_graph) / grid_size).astype(np.int)

    # Initialize things to save
    parameters_save = parameters.copy()
    best_model_state = copy.deepcopy(model.state_dict())
    best_accu = 0
    
    # Get ready to train
    start_time = time.perf_counter()
    model.train()

    # Break the list up into smaller batches for more efficient training
    inputMiniBatches = torch.split(inputs_train, parameters['batch_size'])
    outputMiniBatches = torch.split(outputs_train, parameters['batch_size'])
    numMiniBatch = len(inputMiniBatches)
    
    testingMiniBatches = torch.split(inputs_test, parameters['batch_size'])
    numMiniBatchTest = len(testingMiniBatches)

    # Set up the training functions
    lossFunc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=parameters['learning_rate'], weight_decay = parameters['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=parameters['lr_red_factor'], patience=parameters['lr_red_patience'], threshold=parameters['lr_red_threshold'])

    # Initialize graph data
    graph_data = new_graph_data(parameters['n_epochs'])

    # Actually train
    for epoch in range(parameters['n_epochs']):
        # Everything that needs to be done every epoch
        with torch.no_grad():
            model.eval()
            
            # Data for the accuracy curve
            test_prediction_temp = torch.tensor([]).cuda()
            train_prediction_temp = torch.tensor([]).cuda()
 
            #Baches to minibatch the testing data
            idx = torch.randperm(torch.numel(outputs_test)) 
            inputMiniBatchesTest = torch.split(inputs_test[idx], parameters['batch_size'])
            inputMiniBatchesRep = torch.split(inputs_train_rep[idx], parameters['batch_size']) # Only works because picked representative training set to be of the same size as the testing set

            #Puts the testing output data in the same order as the new minibatches input data
            outputs_test_np_batched = outputs_test_np[idx]
            outputs_train_rep_batched = outputs_train_rep[idx]
            outputs_test_batched = outputs_test[idx]
           
            for minibatch in range(numMiniBatchTest):
                #Generates the model's prediction of the testing data
                test_prediction_temp = torch.cat((test_prediction_temp, model(inputMiniBatchesTest[minibatch])),axis=0)

                # Data for the other plots
                train_prediction_temp = torch.cat((train_prediction_temp, model(inputMiniBatchesRep[minibatch])),axis=0) # To speed up only evaluate the representative training set
            
            #These starts putting the caculcated data into their respective graphing groups                
            score_temp = v_accu_test(test_prediction_temp.cpu().detach().numpy().flatten(), outputs_test_np_batched)
            accu_temp = np.sum(score_temp) / total_num
            graph_data['accu_vals'][epoch] = accu_temp 
            graph_data['accu_epochs'][epoch] = epoch 

            # Data for accu_out
            np.add.at(grid_accu_tally, (grid_num, epoch, 0), score_temp)
            np.add.at(grid_accu_tally, (grid_num, epoch, 1), 1)

            train_loss_temp = lossFunc(train_prediction_temp, outputs_train_rep_batched).item()
            test_loss_temp = lossFunc(test_prediction_temp, outputs_test_batched).item() 
            graph_data['train_loss_vals'][epoch] = train_loss_temp
            graph_data['train_loss_epochs'][epoch] = epoch
            graph_data['test_loss_vals'][epoch] = test_loss_temp
            graph_data['test_loss_epochs'][epoch] = epoch
            graph_data['time_vals'][epoch] = time.perf_counter() - start_time
            graph_data['time_epochs'][epoch] = epoch
        
        # Save the best model
        if accu_temp > best_accu:
            best_accu = accu_temp
            best_model_state = copy.deepcopy(model.state_dict())
            parameters_save['n_epochs'] = epoch

        # Things that need to be done every 10 epochs
        if epoch%10 == 0:
            if show_progress:
                print('=>Starting {}/{} epochs.'.format(epoch+1,parameters['n_epochs']))
            
            with torch.no_grad():
                model.eval()
                
        model.train()

        # Randomize minibatch selection again
        idx = torch.randperm(torch.numel(outputs_train))
        inputMiniBatches = torch.split(inputs_train[idx], parameters['batch_size'])
        outputMiniBatches = torch.split(outputs_train[idx], parameters['batch_size'])

        for minibatch in range(numMiniBatch):
            prediction = model(inputMiniBatches[minibatch])
            loss = lossFunc(prediction,outputMiniBatches[minibatch])
            for param in model.parameters():
                param.grad = None
            #optimizer.zero_grad(set_to_none=True) # This is slower than the above by experiment.
            loss.backward()
            optimizer.step()
        scheduler.step(test_loss_temp)
    
    # Data for the residual plots
    model.eval()
    
    with torch.no_grad():
        #Minibatching the last test data calculation
        test_prediction_temp_final = torch.tensor([]).cuda()

        idx = torch.randperm(torch.numel(outputs_test))
        inputMiniBatchesTest = torch.split(inputs_test[idx], parameters['batch_size'])
        test_outputs_batched = outputs_test[idx]

        for minibatch in range(numMiniBatchTest):
            test_prediction_temp_final = torch.cat((test_prediction_temp_final, model(inputMiniBatchesTest[minibatch])), axis=0)
        residual = test_outputs_batched - test_prediction_temp_final #The numerical error of the current prediction for each test.
        graph_data['out_residual_vals'] = residual.cpu().detach().numpy().flatten()
    
    # Data for the weights and biases histograms
    model_param  = model.state_dict()
    for key in model_param.keys():
        if 'weight' in key:
            graph_data['weights'] = np.append(graph_data['weights'], model_param[key].cpu().detach().numpy().flatten())
        elif 'bias' in key:
            graph_data['biases'] = np.append(graph_data['biases'], model_param[key].cpu().detach().numpy().flatten())
    # Data for accu_out
    graph_data['accu_out_grid'] = np.linspace(min_graph, max_graph, parameters['accu_out_resolution']+1)
    grid_accu_tally_nonzero = np.where(grid_accu_tally[:,:,1:2] == 0, 1, grid_accu_tally)
    graph_data['accu_out_grid_accu'] = ma.where(grid_accu_tally[:,:,1] == 0, ma.masked, grid_accu_tally[:,:,0] / grid_accu_tally_nonzero[:,:,1]) # Masked array

    # Other data
    graph_data['test_outputs'] = outputs_test_np

    print ('Training done!')
    print ('--- %s seconds ---' % (time.perf_counter() - start_time))
    print('Best accuracy: ', best_accu, ', achieved in ', parameters_save['n_epochs'], 'epochs')
    return (graph_data, best_model_state, parameters_save)

# New graph_data
def new_graph_data(epochs):
    """
    Creates a new data dictionary for graphing later; The graphs are those pertaining to one run of one architecture only.

    Inputs: epochs (integer)

    Outputs: graph_data (dictionary)
    """
    graph_data = {}
    graph_data['test_outputs'] = np.array([])
    graph_data['train_loss_vals'] = np.zeros(epochs)
    graph_data['train_loss_epochs'] = np.zeros(epochs)
    graph_data['test_loss_vals'] = np.zeros(epochs)
    graph_data['test_loss_epochs'] = np.zeros(epochs)
    graph_data['accu_vals'] = np.zeros(epochs)
    graph_data['accu_epochs'] = np.zeros(epochs)
    graph_data['time_vals'] = np.zeros(epochs)
    graph_data['time_epochs'] = np.zeros(epochs)
    graph_data['accu_out_grid'] = np.array([])
    graph_data['accu_out_grid_accu'] = np.array([])
    graph_data['out_residual_vals'] = np.array([])
    graph_data['weights'] = np.array([])
    graph_data['biases'] = np.array([])

    return graph_data

# New analysis_data
def new_analysis_data():
    """
    Creates a new data dictionary to store the performance results of different networks.

    Inputs: None

    Outputs: analysis_data (dictionary)
    """
    analysis_data = {}
    analysis_data['nodes'] = np.array([])
    analysis_data['layers'] = np.array([])
    analysis_data['epochs'] = np.array([])
    analysis_data['time'] = np.array([])
    analysis_data['accuracy'] = np.array([])

    return analysis_data

# New graphs (figs, axes, etc.)
def new_graphs():
    """
    Creates new graphs.

    Inputs: None

    Outputs: graphs (dictionary)
    """
    fig_time, ax_time = plt.subplots()
    fig_param, ax_param = plt.subplots()
    fig_loss, ax_loss = plt.subplots()
    fig_accu, ax_accu = plt.subplots()
    fig_accu_out, (ax_out_freq, ax_accu_out) = plt.subplots(nrows=1, ncols=2)
    fig_out_residual, ax_out_residual = plt.subplots()
    fig_weights, ax_weights = plt.subplots()
    fig_biases, ax_biases = plt.subplots()

    return {'fig_time': fig_time, 'ax_time': ax_time, 'fig_param': fig_param, 'ax_param': ax_param, 'fig_loss': fig_loss, 'ax_loss': ax_loss, 'fig_accu': fig_accu, 'ax_accu': ax_accu, 'fig_accu_out': fig_accu_out, 'ax_out_freq': ax_out_freq, 'ax_accu_out': ax_accu_out, 'fig_out_residual': fig_out_residual, 'ax_out_residual': ax_out_residual, 'fig_weights': fig_weights,'ax_weights': ax_weights,'fig_biases': fig_biases,'ax_biases': ax_biases,}


# Do the graphing
# TODO: Use OOP for graphing
def graphing(graphs, graph_data, parameters):
    """
    Does the graphing.
    Inputs: graphs (dictionary), graph_data (dictionary), parameters (dictionary)
    Outputs: graphs (dictionary)
    """
    test_outputs = graph_data['test_outputs']

    param_str = '\n'.join((
        r'Training Size: %d' % parameters['train_size'],
        r'Testing Size: %d' % parameters['test_size'],
        r'Nodes: %d' % parameters['hidden_nodes'],
        r'Layers: %d' % parameters['hidden_layers'],
        r'Minibatch Size: %d' % parameters['batch_size'],
        r'Epochs: %d' % parameters['n_epochs'],
        r'Initial Learning Rate: %f' % parameters['learning_rate'],
        r'Learning Rate Reduction Factor: %f' % parameters['lr_red_factor'],
        r'Learning Rate Reduction Patience: %d' % parameters['lr_red_patience'],
        r'Learning Rate Reduction Threshold: %f' % parameters['lr_red_threshold'],
        r'Weight Decay: %f' % parameters['weight_decay'],
        r'Polynomial Layer: %f' % parameters['polynomial'],
        r'Polynomial Degree: %f' % parameters['polynomial_degree']))
    props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
    graphs['ax_param'].text(0.05, 0.95, param_str, transform=graphs['ax_param'].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    graphs['fig_param'].tight_layout()

    graphs['ax_time'].plot(graph_data['time_epochs'], graph_data['time_vals'], 'b-')
    graphs['ax_time'].set_xlabel('Epochs')
    graphs['ax_time'].set_ylabel('Time (s)')
    graphs['fig_time'].tight_layout()

    graphs['ax_accu'].plot(graph_data['accu_epochs'], graph_data['accu_vals'], 'b-')
    graphs['ax_accu'].set_xlabel('Epochs')
    graphs['ax_accu'].set_ylabel('Accuracy')
    graphs['fig_accu'].tight_layout()

    im = graphs['ax_accu_out'].pcolormesh(np.arange(-0.5, parameters['n_epochs'], 1), graph_data['accu_out_grid'], graph_data['accu_out_grid_accu'])
    graphs['ax_accu_out'].set_xlabel('Epochs')
    graphs['ax_accu_out'].set_ylabel('Outputs')
    graphs['fig_accu_out'].colorbar(im, ax=graphs['ax_accu_out'], label='Accuracy')

    graphs['ax_out_freq'].hist(test_outputs, bins=graph_data['accu_out_grid'], orientation='horizontal', color='b')
    graphs['ax_out_freq'].set_xlabel('Frequency')
    graphs['ax_out_freq'].set_ylabel('Outputs')
    graphs['ax_out_freq'].margins(0)
    graphs['fig_accu_out'].tight_layout()

    train_loss_line, = graphs['ax_loss'].plot(graph_data['train_loss_epochs'], graph_data['train_loss_vals'], 'b-', linewidth=1)
    test_loss_line, = graphs['ax_loss'].plot(graph_data['test_loss_epochs'], graph_data['test_loss_vals'], 'g-', linewidth=1)
    graphs['ax_loss'].legend([train_loss_line, test_loss_line], ['Train', 'Test'])
    graphs['ax_loss'].set_xlabel('Epochs')
    graphs['ax_loss'].set_ylabel('MSE Loss')
    graphs['ax_loss'].set_yscale('log')
    graphs['fig_loss'].tight_layout()

    h = graphs['ax_out_residual'].hist2d(test_outputs, graph_data['out_residual_vals'], [parameters['out_residual_resolution'], parameters['out_residual_resolution']],norm=colors.LogNorm())
    graphs['ax_out_residual'].set_xlabel('True Outputs')
    graphs['ax_out_residual'].set_ylabel('Residual (actual - prediction)')
    graphs['fig_out_residual'].colorbar(h[3], ax=graphs['ax_out_residual'], label='Frequency')
    graphs['ax_out_residual'].set_xlim(0,100)
    graphs['ax_out_residual'].set_ylim(-2,2)
    graphs['fig_out_residual'].tight_layout()

    w1 = graphs['ax_weights'].boxplot(graph_data['weights'], vert = 0, whis = (5,95),showfliers=False)
    graphs['ax_weights'].set_xlabel('Weights')
    graphs['ax_weights'].title.set_text('Weights')
    
    b1 = graphs['ax_biases'].boxplot(graph_data['biases'], vert = 0,showfliers=False)
    graphs['ax_biases'].set_xlabel('Biases')
    graphs['ax_biases'].title.set_text('Biases')

    return graphs

# Show graphs
def show_graphs(graphs):
    """
    Shows the graphs.

    Inputs: graphs (dictionary)

    Outputs: None
    """
    graphs['fig_time']
    graphs['fig_param']
    graphs['fig_loss']
    graphs['fig_accu']
    graphs['fig_accu_out']
    graphs['fig_out_residual']
    graphs['fig_weights']
    graphs['fig_biases']

# Save all graphs in one pdf file
def save_graphs(graphs, graph_data, name):
    """
    Saves the graphs to one pdf.

    Inputs: graphs (dictionary), graph_data (dictionary), name (string)

    Outputs: None
    """
    pp = PdfPages(f'{name}_train.pdf')
    pp.savefig(graphs['fig_time'])
    pp.savefig(graphs['fig_param'])
    pp.savefig(graphs['fig_loss'])
    pp.savefig(graphs['fig_accu'])
    pp.savefig(graphs['fig_accu_out'])
    pp.savefig(graphs['fig_out_residual'])
    pp.savefig(graphs['fig_weights'])
    pp.savefig(graphs['fig_biases'])
    pp.close()

    np.savez(f'{name}_train.npz', **graph_data)

# TODO (code): Implement some sort of autotune
# Profiled scans
def profile(model, inputs_raw, idx_list, profile_parameters):
    '''
    Makes profiled likelihood scans. For given values of some WCs, finds the values of the other WCs that minimizes the NLL.

    Inputs:
        model: The trained DNN with all layers built in
        inputs_raw (np array): The values of the input WCs to be scanned over. The values of the other
          WCs (to be profiled over) don't matter.
        idx_list: A list of the indices of the WCs to be scanned.
        profile_parameters: A dictionary of the profiling hyperparameters, similar to DNN 
        training:
            batch_size
            epochs
            learning_rate
            lr_red_factor
            lr_red_patience
            lr_red_threshold
            rand_pts: The number of random starting points.
            rand_stdev: The width of the distribution of the random starting points.
                currently it's a uniform distribution, so this represents the width.

    Outputs: A tuple of the following tensors:
        min_WCs_scanned: The WC values profiled
        outputs: The profiled likelihood values
    '''

    batch_size = profile_parameters['batch_size']
    epochs = profile_parameters['epochs']
    learning_rate = profile_parameters['learning_rate']
    lr_red_factor = profile_parameters['lr_red_factor']
    lr_red_patience = profile_parameters['lr_red_patience']
    lr_red_threshold = profile_parameters['lr_red_threshold']
    rand_pts = profile_parameters['rand_pts']
    rand_stdev = profile_parameters['rand_stdev']

    # TODO (code): Rewrite this segment in torch instead of np
    inputs_shape = np.insert(np.array(inputs_raw.shape), -1, rand_pts)
    inputs = (np.random.random_sample(inputs_shape) - 0.5) * rand_stdev
    # copy over the WCs being scanned, while leaving the other 14 randomized
    for idx in idx_list:
        inputs[..., idx] = inputs_raw[..., np.newaxis, idx] # broadcasts into the random starting point axis
        # TODO (code): Try randomizing inputs_raw before doing anything. Currently, the 
        # points with similar likelihoods are always profiled together, which may not have 
        # the same optimal hyperparameters.
    inputs = torch.from_numpy(inputs).float().cuda()
    inputs.requires_grad = True
    min_WCs_scanned = torch.full(inputs_raw.shape, torch.nan) # The optimized input WCs
    outputs = torch.full((inputs_shape[0], 1), 10000.).cuda() # fill outputs with 10000. as a default (decimal to force dtype=float not int)
    inputMiniBatches = torch.split(inputs, batch_size)
    batch_idx = 0

    start_time = time.perf_counter()

     # Memory debugging info
    '''
    memory_fig, memory_ax = plt.subplots()
    memory_count = 0
    memory_count_arr = np.full((len(inputMiniBatches)*epochs), np.NaN)
    memory_vals = np.full((len(inputMiniBatches)*epochs), np.NaN)
    '''

    for minibatch in range(len(inputMiniBatches)):
        optimizer = torch.optim.Adam([inputs],lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_red_factor, patience=lr_red_patience, threshold=lr_red_threshold)
        inputMiniBatches = torch.split(inputs, batch_size)
        batch_inputs = inputMiniBatches[minibatch]
        min_outputs = model(batch_inputs)
        min_WCs = batch_inputs.detach().clone() # A snapshot of the WCs of all the points to scan
        optimizer.zero_grad()

        info_string = 'WC indices: '
        for idx in idx_list:
            info_string += f'{idx} '
        info_string += f'\nStarting {minibatch}/{len(inputMiniBatches)} minibatches.'
        print (info_string)

        for epoch in range(epochs):
            print (f'Starting {epoch}/{epochs} epochs of {minibatch}/{len(inputMiniBatches)} minibatches.')

            inputMiniBatches = torch.split(inputs, batch_size)
            batch_inputs = inputMiniBatches[minibatch]
            batch_outputs = model(batch_inputs)
            batch_outputs_cp = batch_outputs.detach().clone()
            batch_outputs_sum = torch.sum(torch.log(batch_outputs + 10)) # Optimize the sum of outputs + 10, equivalent to optimize each output
            idx_to_update = torch.where(batch_outputs_cp < min_outputs)[0]
            min_outputs[idx_to_update] = batch_outputs_cp[idx_to_update]
            min_WCs[idx_to_update] = batch_inputs.detach().clone()[idx_to_update]
            optimizer.zero_grad()
            batch_outputs_sum.backward()
            #print (torch.count_nonzero(inputs.grad))
            #print (inputs.grad)
            #print (inputs.grad[1500,31,6])
            print (f'Sum(log(minibatch outputs+10)): {batch_outputs_sum}')
            #print (f'Output of a random point: {batch_outputs[500,0,0]}')
            for idx in idx_list:
                inputs.grad[..., idx] = 0
            optimizer.step()
            scheduler.step(batch_outputs_sum) # learning rate reduction

            '''
            # Memory debugging info
            memory_count_arr[memory_count] = memory_count
            memory_vals[memory_count] = torch.cuda.memory_allocated()/1e9
            memory_ax.cla()
            memory_ax.plot(memory_count_arr, memory_vals)
            memory_count += 1
            '''

        (min_outputs_scanned, min_starting_point_indicies) = torch.min(min_outputs, -2) # Get the best starting points

        # Make the index tensor suitable for gathering the min_WCs
        min_starting_point_indicies = min_starting_point_indicies.unsqueeze(-2)
        min_starting_point_indicies_shape = list(min_starting_point_indicies.shape)
        min_starting_point_indicies_shape[-1] = inputs_raw.shape[-1]
        min_starting_point_indicies = min_starting_point_indicies.expand(min_starting_point_indicies_shape) # Warning: don't perform in-place operations on this since expand() does not allocate new memory

        min_WCs_scanned[batch_idx: batch_idx + batch_outputs.shape[0]] = torch.squeeze(torch.gather(min_WCs, -2, min_starting_point_indicies), -2).detach().clone() # Get the WCs corresponding to the best-performing starting points
        outputs[batch_idx: batch_idx + batch_outputs.shape[0]] = min_outputs_scanned.detach().clone() # detach from graph to delete obsolete graphs from memory! This was the culprit causing the memory leak
        batch_idx += batch_outputs.shape[0]
    print ('Profiling done!')
    print (f'Time used: {time.perf_counter() - start_time} seconds.')
    # These outputs are raw! Multiply by 2 outside
    return (min_WCs_scanned, outputs)

# Analyze NN
# TEST THIS
def analyze(param_list, trials, std_inputs, std_outputs, std_test_inputs, std_test_outputs, output_stats, std_inputs_rep, std_outputs_rep):
    """
    Tests networks with given hyperparameters.
    Inputs: param_list (list), trials (integer), std_inputs (standardized training input data; Pytorch tensor), std_outputs (standardized training output data; Pytorch tensor), std_test_inputs (standardized testing input data; Pytorch tensor), std_test_outputs (standardized testing output data; Pytorch tensor), output_stats (mean, standard deviation) (tuple), std_inputs_rep (representative inputs; Pytorch tensor), std_outputs_rep (representative outputs; Pytorch tensor)
    Outputs: analysis_data (dictionary)
    """
    analysis_data = new_analysis_data()
    for i in param_list:
        for j in range(trials):
            model = create_model(std_inputs, std_outputs, i)
            graph_data = train_network(model, std_inputs, std_outputs, std_test_inputs, std_test_outputs, output_stats, std_inputs_rep, std_outputs_rep, i, False)
            analysis_data['nodes'] = np.append(analysis_data['nodes'], np.full(i['n_epochs'], i['hidden_nodes']))
            analysis_data['layers'] = np.append(analysis_data['layers'], np.full(i['n_epochs'], i['hidden_layers']))
            analysis_data['epochs'] = np.append(analysis_data['epochs'], graph_data['accu_epochs'])
            analysis_data['time'] = np.append(analysis_data['time'], graph_data['time_vals'])
            analysis_data['accuracy'] = np.append(analysis_data['accuracy'], graph_data['accu_vals'])
    return analysis_data

# New analysis graphs (figs, axes, etc.)
# TEST THIS
def new_analysis_graphs():
    """
    Creates new analysis graphs.
    Inputs: None
    Outputs: analysis_graphs (dictionary)
    """
    fig_analysis, ax_analysis = plt.subplots()

    return {'fig_analysis': fig_analysis, 'ax_analysis': ax_analysis}

# Graph the data from NN analysis
# TEST THIS
def analysis_graphing(analysis_graphs, analysis_data, param_list, trials):
    """
    Does the analysis graphing.
    Inputs: analysis_graphs (dictionary), analysis_data (dictionary), param_list (list), trials (integer)
    Outputs: analysis_graphs (dictionary)
    """
    analysis_graphs['ax_analysis'].set_xlabel('Time (s)')
    analysis_graphs['ax_analysis'].set_ylabel('Accuracy')
    counter = 0
    for i in param_list:
        first_line = analysis_graphs['ax_analysis'].plot(analysis_data['time'][counter: counter+i['n_epochs']], analysis_data['accuracy'][counter: counter+i['n_epochs']], label=str(i))
        counter += i['n_epochs']
        for j in range(trials - 1):
            analysis_graphs['ax_analysis'].plot(analysis_data['time'][counter: counter+i['n_epochs']], analysis_data['accuracy'][counter: counter+i['n_epochs']], color=first_line[0].get_color())
            counter += i['n_epochs']
    analysis_graphs['ax_analysis'].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    analysis_graphs['fig_analysis'].tight_layout()

    return analysis_graphs

# Classes

# Affine Layer (equivalent to nn.functional.Linear inside a Module)
class LinearLayer(nn.Module):
    """Outputs an affine transformation or its inverse of inputs.

    y = x * A^T + b

    Attributes:
       A: The matrix to be applied to the input
       b: The constant to be added to the input

    """
    def __init__(self, A: torch.Tensor, b: torch.Tensor) -> None:
        """Constructor

        Args:
            A: The matrix to be applied to the input
            b: The constant to be added to the input
        """
        
        super(LinearLayer,self).__init__()

        self.A = A.cuda()
        self.b = b.cuda()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Performs forward propagation for this module

        Args: 
            x: 
              The input tensor to this layer. This function preserves
              all indices except the last one, which is assumed to
              index the input variables. Leading indices can be used
              for minibatches or structuring the inputs.

        """
        
        if x.shape[-1] != self.A.shape[1]:
            raise IndexError(f'Expecting {x.shape[-1]} inputs for matrix, got {self.A.shape[1]}')
        return nn.functional.linear(x, self.A, self.b)
    
# General Differentiable Layer
# Warning: the function passed in needs to deal with all the indicies of the input, and is in general not easy to vectorize.
class GeneralLayer(nn.Module):
    """Outputs an arbitrary differentiable function of inputs.

    Attributes:
       n_inputs: The number of inputs this layer expects
       fcn: The differentiable function to be applied to the inputs

    """
    def __init__(self, n_inputs: int, fcn: callable) -> None:
        """Constructor

        Args:
            n_inputs: The number of inputs this layer expects
            fcn: The differentiable function to be applied to the inputs
        """
        
        super(GeneralLayer,self).__init__()

        self.n_inputs = n_inputs
        self.fcn = fcn

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Performs forward propagation for this module

        Args: 
            x: 
              The input tensor to this layer. This function does NOT
              preserve any indices. It merely applies the specified
              function to the input.

        """
        
        if x.shape[-1] != self.n_inputs:
            raise IndexError(f'Expecting {self.n_inputs} inputs, got {x.shape[-1]}')
        return self.fcn(x)