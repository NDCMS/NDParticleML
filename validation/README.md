# Validation

This folder contains current NN validation code.
- `2d_profiling.sh`, `2d_profiling.submit`, and `2d_profiling.py` are to be used together to perform 2D validation with self-consistent profiling. This means that for each value of the WC being scanned, the other 15 WCs are fit to produce the minimum $2\Delta NLL$.
- `1d_profiling.sh`, `1d_profiling.submit`, and `1d_profiling.py` are to be used together to perform 1D validation with self-consistent profiling.
- `validation.ipynb`: 1D validation without self-consistent profiling. This means the profiled scans merely evaluate the NN at the same points in 16D space as the `Combine` scans.
- `validation_2d.ipynb`: 2D validation without self-consistent profiling.
- `validation_2d_profiling.ipynb`: 2D validation with self-consistent profiling. The same as `2d_profiling.py`, just in a notebook.
- `comparison_plots.ipynb`: Compares 1D validation without self-consistent profiling accuracy of two models.
- `graphs`: Stores the graphs for the validation.

There is currently no script to do 1D validation with self-consistent profiling, although it is easy to create by adapting that in v1.

Note: The current `Combine` scan data is in `/tmpscratch/sliu24/demos/`.
