# Validation

This folder contains current NN validation code.
- `2d_profiling.sh`, `2d_profiling.submit`, and `2d_profiling.py` are to be used together to perform 2D validation with self-consistent profiling. This means that for each value of the WC being scanned, the other 15 WCs are fit to produce the minimum $2\Delta NLL$.
- `1d_profiling.sh`, `1d_profiling.submit`, and `1d_profiling.py` are to be used together to perform 1D validation with self-consistent profiling.
- `validation.ipynb`: 1D validation without self-consistent profiling. This means the profiled scans merely evaluate the NN at the same points in 16D space as the `Combine` scans.
- `validation_2d.ipynb`: 2D validation without self-consistent profiling.
- `validation_2d_profiling.ipynb`: 2D validation with self-consistent profiling. The same as `2d_profiling.py`, just in a notebook.
- `comparison_plots.ipynb`: Compares 1D validation without self-consistent profiling accuracy of two models. Needs to be updated.
- `graphs`: Stores the graphs for the validation.

There is currently no script to do 1D validation with self-consistent profiling, although it is easy to create by adapting that in v1.

## Notes

- The current `Combine` scan data is in `/tmpscratch/sliu24/demos/`.
- There is an unknown constant shift between the 7M dataset (probably lost), the 50M dataset (`/scratch365/klannon/dnnlikelihood/likelihood_data_no_delta.npz`), the frozen `combine` scans, and the profiled `combine` scans.
    - The frozen scans seem to all be shifted so that their respective minima are 0.
    - The profiled scans (as of 4/3/2023) share the same shift as the 7M dataset.
    - The 50M dataset is shifted by an unknown amount relative to the 7M dataset.
    - To recover the unshifted profiled scan comparisons, I took the difference between the minima of the target and model profiled scans of `ctp`, because these were shown to match almost exactly (no false minimum issue) using the `3169_0` model. model - target = **0.34523553**.