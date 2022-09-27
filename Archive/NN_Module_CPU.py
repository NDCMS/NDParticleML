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
    return abs(pred-act)
def rel_err(pred, act):
    return abs(abs_err(pred, act)/act)

# Check what proportion of the predictions falls within 0.01 absolute accuracy or 1% relative accuracy
def accu_test(prediction, actual):
    "Output 1 if prediction is accurate enough and 0 otherwise"
    if (abs_err(prediction, actual) < 0.01 or rel_err(prediction, actual) < 0.01):
        return 1
    else: return 0
v_accu_test = np.vectorize(accu_test) # This makes a vector function for convenience

# Create a simple neural network with layer and node variabliltiy
def create_model(inputs, outputs, hidden_nodes=100, layer_num = 0):
    layers = [torch.nn.Linear(inputs.shape[1],hidden_nodes),torch.nn.ReLU()]
    for i in range(layer_num):
        layers.append(torch.nn.Linear(hidden_nodes,hidden_nodes))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(hidden_nodes,1)) # We only care about functions with one output
    model = torch.nn.Sequential(*layers)
    # include different number of nodes per layer functionality
    #list with nodes per layer
    return (model, hidden_nodes, layer_num)