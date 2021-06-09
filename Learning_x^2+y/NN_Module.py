# %matplotlib notebook
import torch
import numpy as np
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

# Check what proportion of the predictions falls within 0.01 absolute accuracy or 1% relative accuracy
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