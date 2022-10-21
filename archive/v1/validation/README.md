# Validation

This folder contains NN validation code at the end of summer 2021.
- `pytorch_profiling.sh` and `pytorch_profiling.submit` are to be used with scripts in `../demos/pytorch_profiling`. These perform 1D validation with self-consistent profiling. This means that for each value of the WC being scanned, the other 15 WCs are fit to produce the minimum $2\Delta NLL$.
- `Validation.ipynb`: 1D validation without self-consistent profiling. This means the profiled scans merely evaluate the NN at the same points in 16D space as the `Combine` scans.
- `Validation_2D.ipynb`: 2D validation without self-consistent profiling.
- `Validation_2D_with_profiling.ipynb`: 2D validation with self-consistent profiling.