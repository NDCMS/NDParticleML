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
    return abs(abs_err(pred, act)/act)

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
def create_model(inputs, outputs, hidden_nodes=100, layer_num = 0):
    """
    Creates a sequential model with the same number of nodes in each hidden layer.

    Inputs: inputs (Pytorch tensor), outputs (Pytorch tensor), hidden_nodes (integer), layer_num (integer)

    Outputs: model (Pytorch sequential container)
    """
    layers = [torch.nn.Linear(inputs.shape[1],hidden_nodes),torch.nn.ReLU()]
    for i in range(layer_num):
        layers.append(torch.nn.Linear(hidden_nodes,hidden_nodes))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(hidden_nodes,1)) # We only care about functions with one output
    model = torch.nn.Sequential(*layers)
    # include different number of nodes per layer functionality
    #list with nodes per layer
    return model.cuda()

# Train network
def train_network(model, hidden_nodes, hidden_layers, inputs, outputs, test_inputs, test_outputs, miniBatchSize = 100., num_epochs = 500, learning_rate = 1e-4, weight_decay = 1e-5, show_progress = True):
    """
    Trains a network of a given architecture.

    Inputs: model (Pytorch sequential container), hidden_nodes (the number of nodes in each hidden layer (the same for all layers); integer), hidden_layers (integer), inputs (training input data; Pytorch tensor), outputs (training output data; Pytorch tensor), test_inputs (testing input data; Pytorch tensor), test_outputs (testing output data; Pytorch tensor), analysis_data (dictionary), miniBatchSize (integer), num_epochs (integer), show_progress (boolean)

    Outputs: graph_data (dictionary)
    """
    # Useful information
    total_num = torch.numel(test_outputs)
    N = inputs.shape[1]

    # Get ready to train
    start_time = time.perf_counter()
    model.train()

    # Break the list up into smaller batches for more efficient training
    numMiniBatch = int(math.floor(inputs.shape[0]/miniBatchSize))
    inputMiniBatches = inputs.chunk(numMiniBatch)
    outputMiniBatches = outputs.chunk(numMiniBatch)

    # Set up the training functions
    lossFunc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay = weight_decay)

    # Initialize graph data
    graph_data = new_graph_data(total_num, num_epochs)

    # Actually train
    for epoch in range(num_epochs):
        # Everything that needs to be done every epoch
        with torch.no_grad():
            model.eval()
            # Data for the accuracy plots
            prediction_temp = model(test_inputs)
            score_temp = v_accu_test(prediction_temp.cpu().detach().numpy(), test_outputs.cpu().detach().numpy())
            graph_data['accu_vals'][epoch] = np.sum(score_temp) / total_num
            graph_data['accu_epochs'][epoch] = epoch
            for i in range(total_num):
                if (score_temp[i] == 1):
                    graph_data['accu_out_colors'][i + epoch*total_num] = 'b'
                graph_data['accu_out_epochs'][i + epoch*total_num] = epoch
                graph_data['accu_out_vals'][i + epoch*total_num] = test_outputs[i]

            # Data for the other plots
            train_total_prediction_temp = model(inputs)
            train_total_loss_temp = lossFunc(train_total_prediction_temp, outputs).item()
            test_total_prediction_temp = model(test_inputs)
            test_total_loss_temp = lossFunc(test_total_prediction_temp, test_outputs).item()
            graph_data['train_loss_vals'][epoch] = train_total_loss_temp
            graph_data['train_loss_epochs'][epoch] = epoch
            graph_data['test_loss_vals'][epoch] = test_total_loss_temp
            graph_data['test_loss_epochs'][epoch] = epoch
            graph_data['time_vals'][epoch] = time.perf_counter() - start_time
            graph_data['time_epochs'][epoch] = epoch

        # Things that need to be done every 10 epochs
        if epoch%10 == 0:
            if show_progress:
                print('=>Starting {}/{} epochs.'.format(epoch+1,num_epochs))
            
            with torch.no_grad():
                model.eval()
                
        model.train()
                
        for minibatch in range(numMiniBatch):
            prediction = model(inputMiniBatches[minibatch])
            loss = lossFunc(prediction,outputMiniBatches[minibatch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Data for the residual plots
    model.eval()
    prediction_temp = model(test_inputs)
    residual = test_outputs - prediction_temp
    graph_data['out_residual_vals'] = np.append(graph_data['out_residual_vals'], residual.cpu().detach().numpy())
    
    # Data for the weights and biases histograms
    model_param  = model.state_dict()
    for key in model_param.keys():
        if 'weight' in key:
            graph_data['weights'] = np.append(graph_data['weights'], model_param[key].cpu().detach().numpy().flatten())
        elif 'bias' in key:
            graph_data['biases'] = np.append(graph_data['biases'], model_param[key].cpu().detach().numpy().flatten())
    # Other data
    graph_data['test_outputs'] = test_outputs

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
    graph_data['accu_out_vals'] = np.zeros(epochs * total_num)
    graph_data['accu_out_colors'] = np.full(epochs * total_num, 'r')
    graph_data['accu_out_epochs'] = np.zeros(epochs * total_num)
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
    fig_loss, ax_loss = plt.subplots()
    fig_accu, ax_accu = plt.subplots()
    fig_accu_out, (ax_out_freq, ax_accu_out) = plt.subplots(nrows=1, ncols=2)
    fig_out_residual, ax_out_residual = plt.subplots()
    fig_histograms, (ax_weights, ax_biases) = plt.subplots(nrows=1, ncols=2)

    return {'fig_loss': fig_loss, 'ax_loss': ax_loss, 'fig_accu': fig_accu, 'ax_accu': ax_accu, 'fig_accu_out': fig_accu_out, 'ax_out_freq': ax_out_freq, 'ax_accu_out': ax_accu_out, 'fig_out_residual': fig_out_residual, 'ax_out_residual': ax_out_residual, 'fig_histograms': fig_histograms, 'ax_weights': ax_weights, 'ax_biases': ax_biases}

# Do the graphing
def graphing(graphs, graph_data, total_num, epochs, accu_out_resolution=100, out_residual_resolution=100):
    """
    Does the graphing.

    Inputs: graphs (dictionary), graph_data (dictionary)

    Outputs: graphs (dictionary)
    """
    test_outputs = graph_data['test_outputs'].cpu().detach().numpy().flatten()

    graphs['ax_accu'].plot(graph_data['accu_epochs'], graph_data['accu_vals'], 'b-')
    graphs['ax_accu'].set_xlabel('Epochs')
    graphs['ax_accu'].set_ylabel('Accuracy')

    # Calculate accuracy for each region
    max_out = test_outputs.max()
    min_out = test_outputs.min()
    max_graph = max_out + (max_out - min_out) / 100
    min_graph = min_out - (max_out - min_out) / 100
    grid = np.linspace(min_graph, max_graph, accu_out_resolution+1)
    grid_size = (max_graph - min_graph) / accu_out_resolution
    grid_accu_tally = np.zeros((accu_out_resolution, epochs, 2))
    grid_accu = ma.array(np.zeros((accu_out_resolution, epochs))) # Masked array
    for i in range(epochs):
        for j in range(total_num):
            grid_num = int((graph_data['accu_out_vals'][i*total_num + j] - min_graph) / grid_size)
            if graph_data['accu_out_colors'][i*total_num + j] == 'b':
                grid_accu_tally[grid_num][i][0] += 1
            grid_accu_tally[grid_num][i][1] += 1
    for i in range(accu_out_resolution):
        for j in range(epochs):
            if grid_accu_tally[i][j][1] == 0:
                grid_accu[i, j] = ma.masked # Possible matplotlib bug, [i][j] doesn't work but [i,j] does.
            else:
                grid_accu[i][j] = grid_accu_tally[i][j][0] / grid_accu_tally[i][j][1]
    im = graphs['ax_accu_out'].pcolormesh(np.arange(-0.5, epochs, 1), grid, grid_accu)
    graphs['ax_accu_out'].set_xlabel('Epochs')
    graphs['ax_accu_out'].set_ylabel('Outputs')
    graphs['fig_accu_out'].colorbar(im, ax=graphs['ax_accu_out'], label='Accuracy')
    # graphs['ax_accu_out'].scatter(graph_data['accu_out_epochs'], graph_data['accu_out_vals'], c=graph_data['accu_out_colors'], s=1)
    # graphs['ax_accu_out'].set_xlabel('Epochs')
    # graphs['ax_accu_out'].set_ylabel('Outputs (Red=inaccurate, Blue=accurate)')

    graphs['ax_out_freq'].hist(test_outputs, bins=grid, orientation='horizontal', color='b')
    graphs['ax_out_freq'].set_xlabel('Outputs')
    graphs['ax_out_freq'].set_ylabel('Frequency')
    graphs['ax_out_freq'].margins(0)
    graphs['fig_accu_out'].tight_layout()

    train_loss_line, = graphs['ax_loss'].plot(graph_data['train_loss_epochs'], graph_data['train_loss_vals'], 'b-', linewidth=1)
    test_loss_line, = graphs['ax_loss'].plot(graph_data['test_loss_epochs'], graph_data['test_loss_vals'], 'g-', linewidth=1)
    graphs['ax_loss'].legend([train_loss_line, test_loss_line], ['Train', 'Test'])
    graphs['ax_loss'].set_xlabel('Epochs')
    graphs['ax_loss'].set_ylabel('Loss (MSE)')
    graphs['ax_loss'].set_yscale('log')

    h = graphs['ax_out_residual'].hist2d(test_outputs, graph_data['out_residual_vals'], [out_residual_resolution, out_residual_resolution])
    graphs['ax_out_residual'].set_xlabel('True Outputs')
    graphs['ax_out_residual'].set_ylabel('Residual (actual - prediction)')
    graphs['fig_out_residual'].colorbar(h[3], ax=graphs['ax_out_residual'], label='Frequency')

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
    graphs['fig_loss']
    graphs['fig_accu']
    graphs['fig_accu_out']
    graphs['fig_out_residual']
    graphs['fig_histograms']

# Analyze NN
def analyze(param_list, trials, inputs, outputs, test_inputs, test_outputs, miniBatchSize = 100.):
    """
    Tests networks with given hyperparameters.

    Inputs: param_list (list), trials (integer), inputs (training input data; Pytorch tensor), outputs (training output data; Pytorch tensor), test_inputs (testing input data; Pytorch tensor), test_outputs (testing output data; Pytorch tensor), analysis_data (dictionary), miniBatchSize (integer)

    Outputs: analysis_data (dictionary)
    """
    analysis_data = new_analysis_data()
    for i in param_list:
        for j in range(trials):
            model = create_model(inputs, outputs, i[0], i[1])
            graph_data = train_network(model, i[0], i[1], inputs, outputs, test_inputs, test_outputs, miniBatchSize, i[2], i[3], i[4], False)
            analysis_data['nodes'] = np.append(analysis_data['nodes'], np.full(i[2], i[0]))
            analysis_data['layers'] = np.append(analysis_data['layers'], np.full(i[2], i[1]))
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

    Inputs: analysis_graphs (dictionary), analysis_data (dictionary)

    Outputs: analysis_graphs (dictionary)
    """
    analysis_graphs['ax_analysis'].set_xlabel('Time (s)')
    analysis_graphs['ax_analysis'].set_ylabel('Accuracy')
    counter = 0
    for i in param_list:
        first_line = analysis_graphs['ax_analysis'].plot(analysis_data['time'][counter: counter+i[2]], analysis_data['accuracy'][counter: counter+i[2]], label=str(i))
        counter += i[2]
        for j in range(trials - 1):
            analysis_graphs['ax_analysis'].plot(analysis_data['time'][counter: counter+i[2]], analysis_data['accuracy'][counter: counter+i[2]], color=first_line[0].get_color())
            counter += i[2]
    analysis_graphs['ax_analysis'].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    analysis_graphs['fig_analysis'].tight_layout()

    return analysis_graphs