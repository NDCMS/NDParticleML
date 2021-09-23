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

# Load data sets
loaded_data_50M = np.load('likelihood_data_50M.npz')
use_it_50M = (loaded_data_50M['deltaNLL'] != 0) & (loaded_data_50M['deltaNLL'] < 5) # Only keep the outputs < 5
use_it_50M[0] = True # Keep the first of the repeated entries
loaded_data_5M = np.load('likelihood_data.npz')
use_it_5M = (loaded_data_5M['deltaNLL'] != 0)
use_it_5M[0] = True # Keep the first of the repeated entries

to_save = {}
for key in loaded_data_50M.keys():
    to_save[key] = np.concatenate([loaded_data_5M[key][use_it_5M], loaded_data_50M[key][use_it_50M]], axis=0)

np.savez('likelihood_data_processed.npz', **to_save)