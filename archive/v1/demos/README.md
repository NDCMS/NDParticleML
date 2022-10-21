# Demo Code

This folder contains code useful for proofs of concept.
- `batch`: These scripts demonstrate how to submit batch jobs on CAML. They generate their own data, so are self-contained.
- `pytorch_profiling`: These scripts demonstrate optimization of trained NNs as functions of their inputs. In other words, they find the inputs of the NN which produce the minimum output.
    - `Autograd_Test.ipynb`: Shows how `Autograd` can be used to compute gradients on the NNs inputs instead of the common purpose of NN weights and biases.
    - `contour_test.ipynb`: Makes contour plots given irregular (non-grid) data.
    - `optim_point_by_point.py`: Makes 1D profiled scans point-by-point.
    - `optim_sum_1_starting_point.py`: Makes 1D profiled scans by minimizing the *sum* of the outputs of many points instead of point-by-point.
    - `optim_sum_100_starting_point.py`: Similar to the above, but for each actual point on the graph, uses 100 random starting points and takes the best run (lowest minimum).
- `data_processing.py`: Demonstrates how to select a portion of the training data according to one's needs (e.g., more examples with low outputs).
- `learning_functions.ipynb`: Demonstrates training of NN on custom functions with some hyperparameters.
- `Likelihood_Basic_Notebook.ipynb`: Demonstrates training of NN on real data. A cleaner version of `likelihood.py` for getting started.