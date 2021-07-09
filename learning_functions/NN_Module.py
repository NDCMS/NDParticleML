# %matplotlib notebook
import torch
import numpy as np
import numpy.ma as ma
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import time
import pandas as pd
from pandas import read_csv
from matplotlib.backends.backend_pdf import PdfPages

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

def affine_transform(tensor, stats):
    """
    Subtracts mean and divides by standard deviation.

    Inputs: tensor (Pytorch tensor), stats (mean, standard deviation) (tuple)

    Outputs: tensor (Pytorch tensor)
    """
    return (tensor - stats[0]) / stats[1]

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
    if (abs_err(prediction, actual) < 0.01 or rel_err(prediction, actual) < 0.01):
        return 1
    else: return 0
v_accu_test = np.vectorize(accu_test) # This makes a vector function for convenience

# Create a simple neural network with layer and node variabliltiy
def create_model(inputs, outputs, parameters):
    """
    Creates a sequential model with the same number of nodes in each hidden layer.

    Inputs: inputs (Pytorch tensor), outputs (Pytorch tensor), parameters (dictionary)

    Outputs: model (Pytorch sequential container)
    """
    layers = [torch.nn.Linear(inputs.shape[1],parameters['hidden_nodes']),torch.nn.ReLU()]
    for i in range(parameters['hidden_layers']):
        layers.append(torch.nn.Linear(parameters['hidden_nodes'],parameters['hidden_nodes']))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(parameters['hidden_nodes'],1)) # We only care about functions with one output
    model = torch.nn.Sequential(*layers)
    # include different number of nodes per layer functionality
    #list with nodes per layer
    return model.cuda()

# Train network
def train_network(model, std_inputs, std_outputs, std_test_inputs, std_test_outputs, output_stats, std_inputs_rep, std_outputs_rep, parameters, show_progress = True):
    """
    Trains a network of a given architecture.

    Inputs: model (Pytorch sequential container), hidden_nodes (the number of nodes in each hidden layer (the same for all layers); integer), hidden_layers (integer), std_inputs (standardized training input data; Pytorch tensor), std_outputs (standardized training output data; Pytorch tensor), std_test_inputs (standardized testing input data; Pytorch tensor), std_test_outputs (standardized testing output data; Pytorch tensor), output_stats (mean, standard deviation) (tuple), std_inputs_rep (representative inputs; Pytorch tensor), std_outputs_rep (representative outputs; Pytorch tensor), parameters (dictionary), show_progress (boolean)

    Outputs: graph_data (dictionary)
    """
    # Useful information
    total_num = torch.numel(std_test_outputs)
    N = std_inputs.shape[1]
    test_outputs = affine_untransform(std_test_outputs, output_stats)
    test_outputs_np = test_outputs.cpu().detach().numpy().flatten()
    test_size = test_outputs_np.size

    # Useful info for accu_out graph
    # Calculate accuracy for each region
    max_out = test_outputs_np.max()
    min_out = test_outputs_np.min()
    max_graph = max_out + (max_out - min_out) / 100
    min_graph = min_out - (max_out - min_out) / 100
    grid_size = (max_graph - min_graph) / parameters['accu_out_resolution']
    grid_accu_tally = np.zeros((parameters['accu_out_resolution'], parameters['n_epochs'], 2))
    grid_num = np.floor((test_outputs_np - min_graph) / grid_size).astype(np.int)
    
    # Get ready to train
    start_time = time.perf_counter()
    model.train()

    # Break the list up into smaller batches for more efficient training
    inputMiniBatches = torch.split(std_inputs, parameters['batch_size'])
    outputMiniBatches = torch.split(std_outputs, parameters['batch_size'])
    numMiniBatch = len(inputMiniBatches)

    # Set up the training functions
    lossFunc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=parameters['learning_rate'], weight_decay = parameters['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=parameters['lr_red_factor'], patience=parameters['lr_red_patience'], threshold=parameters['lr_red_threshold'])

    # Initialize graph data
    graph_data = new_graph_data(total_num, parameters['n_epochs'])

    # Actually train
    for epoch in range(parameters['n_epochs']):
        # Everything that needs to be done every epoch
        with torch.no_grad():
            model.eval()
            # Data for the accuracy curve
            test_final_prediction_temp = affine_untransform(model(std_test_inputs), output_stats)
            score_temp = v_accu_test(test_final_prediction_temp.cpu().detach().numpy().flatten(), test_outputs_np)
            graph_data['accu_vals'][epoch] = np.sum(score_temp) / total_num
            graph_data['accu_epochs'][epoch] = epoch

            # Data for accu_out
            np.add.at(grid_accu_tally, (grid_num, epoch, 0), score_temp)
            np.add.at(grid_accu_tally, (grid_num, epoch, 1), 1)

            # Data for the other plots
            train_std_prediction_temp = model(std_inputs_rep) # We only find the loss on a representative sample of the data, with the size equal to that of the testing set, to save memory
            train_std_loss_temp = lossFunc(train_std_prediction_temp, std_outputs_rep).item()
            test_std_prediction_temp = model(std_test_inputs)
            test_std_loss_temp = lossFunc(test_std_prediction_temp, std_test_outputs).item()
            graph_data['train_loss_vals'][epoch] = train_std_loss_temp
            graph_data['train_loss_epochs'][epoch] = epoch
            graph_data['test_loss_vals'][epoch] = test_std_loss_temp
            graph_data['test_loss_epochs'][epoch] = epoch
            graph_data['time_vals'][epoch] = time.perf_counter() - start_time
            graph_data['time_epochs'][epoch] = epoch

        # Things that need to be done every 10 epochs
        if epoch%10 == 0:
            if show_progress:
                print('=>Starting {}/{} epochs.'.format(epoch+1,parameters['n_epochs']))
            
            with torch.no_grad():
                model.eval()
                
        model.train()
                
        for minibatch in range(numMiniBatch):
            prediction = model(inputMiniBatches[minibatch])
            loss = lossFunc(prediction,outputMiniBatches[minibatch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(test_std_loss_temp)
    
    # Data for the residual plots
    model.eval()
    test_final_prediction_temp = affine_untransform(model(std_test_inputs), output_stats)
    residual = test_outputs - test_final_prediction_temp
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
    graph_data['test_outputs'] = test_outputs_np

    print ('Training done!')
    #print ('--- %s seconds ---' % (time.perf_counter() - start_time))
    return graph_data

# New graph_data
def new_graph_data(total_num, epochs):
    """
    Creates a new data dictionary for graphing later; The graphs are those pertaining to one run of one architecture only.

    Inputs: total_num (the total number of testing data points; integer), epochs (integer)

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
    fig_histograms, (ax_weights, ax_biases) = plt.subplots(nrows=1, ncols=2)

    return {'fig_time': fig_time, 'ax_time': ax_time, 'fig_param': fig_param, 'ax_param': ax_param, 'fig_loss': fig_loss, 'ax_loss': ax_loss, 'fig_accu': fig_accu, 'ax_accu': ax_accu, 'fig_accu_out': fig_accu_out, 'ax_out_freq': ax_out_freq, 'ax_accu_out': ax_accu_out, 'fig_out_residual': fig_out_residual, 'ax_out_residual': ax_out_residual, 'fig_histograms': fig_histograms, 'ax_weights': ax_weights, 'ax_biases': ax_biases}

# Do the graphing
def graphing(graphs, graph_data, parameters):
    """
    Does the graphing.

    Inputs: graphs (dictionary), graph_data (dictionary), parameters (dictionary)

    Outputs: graphs (dictionary)
    """
    test_outputs = graph_data['test_outputs']

    param_str = '\n'.join((
        r'Training Size: %d' % parameters['train_size'],
        r'Validation Size: %d' % parameters['test_size'],
        r'Nodes: %d' % parameters['hidden_nodes'],
        r'Layers: %d' % parameters['hidden_layers'],
        r'Minibatch Size: %d' % parameters['batch_size'],
        r'Epochs: %d' % parameters['n_epochs'],
        r'Initial Learning Rate: %f' % parameters['learning_rate'],
        r'Learning Rate Reduction Factor: %f' % parameters['lr_red_factor'],
        r'Learning Rate Reduction Patience: %d' % parameters['lr_red_patience'],
        r'Learning Rate Reduction Threshold: %f' % parameters['lr_red_threshold'],
        r'Weight Decay: %f' % parameters['weight_decay']))
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
    graphs['ax_loss'].set_ylabel('Loss (After Standardization; MSE)')
    graphs['ax_loss'].set_yscale('log')
    graphs['fig_loss'].tight_layout()

    h = graphs['ax_out_residual'].hist2d(test_outputs, graph_data['out_residual_vals'], [parameters['out_residual_resolution'], parameters['out_residual_resolution']])
    graphs['ax_out_residual'].set_xlabel('True Outputs')
    graphs['ax_out_residual'].set_ylabel('Residual (actual - prediction)')
    graphs['fig_out_residual'].colorbar(h[3], ax=graphs['ax_out_residual'], label='Frequency')
    graphs['fig_out_residual'].tight_layout()

    graphs['ax_weights'].hist(graph_data['weights'], color='b')
    graphs['ax_weights'].set_xlabel('Weights')
    graphs['ax_weights'].set_ylabel('Frequency')

    graphs['ax_biases'].hist(graph_data['biases'], color='b')
    graphs['ax_biases'].set_xlabel('Biases')
    graphs['ax_biases'].set_ylabel('Frequency')
    graphs['fig_histograms'].tight_layout()

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
    graphs['fig_histograms']

# Save all graphs in one pdf file
def save_graphs(graphs, name):
    """
    Saves the graphs to one pdf.

    Inputs: graphs (dictionary), name (string)

    Outputs: None
    """
    pp = PdfPages(name)
    pp.savefig(graphs['fig_time'])
    pp.savefig(graphs['fig_param'])
    pp.savefig(graphs['fig_loss'])
    pp.savefig(graphs['fig_accu'])
    pp.savefig(graphs['fig_accu_out'])
    pp.savefig(graphs['fig_out_residual'])
    pp.savefig(graphs['fig_histograms'])
    pp.close()

# Analyze NN
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
def new_analysis_graphs():
    """
    Creates new analysis graphs.

    Inputs: None

    Outputs: analysis_graphs (dictionary)
    """
    fig_analysis, ax_analysis = plt.subplots()

    return {'fig_analysis': fig_analysis, 'ax_analysis': ax_analysis}

# Graph the data from NN analysis
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