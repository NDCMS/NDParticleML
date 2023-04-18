# NDParticleML

This code seeks to train neural networks to learn likelihood functions (LFs) in the Standard Model effective field theory framework calculated from simulated yields parameterized by Wilson Coefficients and observed yields in the CMS detector.

## Overview

The workflow consists of three steps: training, validation, and analysis. Training and validation are completed (although still being improved), and analysis is still in its early stages.

<br/>

**Training**: During this step, a neural network (NN) is trained on a sampling of the LF from the LHC CMS experiment to approximate the LF with sufficient accuracy.
- Requirements
    - A `combine` sampling of the LF
- Products
    - A trained NN as a `.pt` file

<br/>

**Validation**: During this step, the trained NN is tested against correct data—`combine` scans—to analyze its accuracy.
- Requirements
    - `combine` low-dimentional scans of the LF
    - A trained NN
- Products
    - Comparison graphs between `combine` scans and NN scans

<br/>

**Analysis**: During this step, we take the trained NN as the LF and explore the 16D parameter space, taking advantage of the speedup over `combine` samplings.
- Requirements
    - A trained NN
- Products
    - Scans over linear combinations of WCs
    - TBD

<br/>

## Getting Started

This section will show you how to train and validate a NN in our framework. **Warning: For non-NDCMS members, data files need to be obtained via alternative means.**

### Set up the environment

For NDCMS members, [this Google Doc](https://docs.google.com/document/d/17Ql04YOSVGH9-EYsTJp182Pqw5IB1Lnf1sdQ2B77kgI/edit?usp=sharing) is a good reference for setting up CRC and the CAML GPU cluster. Key steps:
- Get a CRC account
- Find your personal CRC directory, including your `scratch365` space.
- Be able to log into CAML's Jupyterhub at https://camlnd.crc.nd.edu:9800/hub/home.

For general use:
- Make sure to run everything on `CUDA`.
- Have `Pytorch` installed, version `>=1.9`.

### Train an example NN

Via batch system:
- Copy the following into your working directory
    - `./archive/v1/training/likelihood.py`
    - `./archive/v1/training/likelihood.sh`
    - `./archive/v1/training/likelihood.submit`
    - `./archive/v1/modules/nn_module_v1.py`
    - `/tmpscratch/sliu24/demos/likelihood_data_processed.npz`
- Check if the `import` statement has the right `nn_module` name
- Run `condor_submit likelihood.submit`
- After finished, there will be graphs and the trained model in their respective folder.

Via Jupyter Notebook:
- Move the contents of `./archive/v1/training/likelihood.py` into a notebook
- Near the bottom, change how the graphs and model are saved. For example, `f'./graphs/{args.out_file}.pdf'` becomes `f'{args.out_file}.pdf'`.
- Copy the following into your working directory
    - The Jupyter Notebook
    - `./archive/v1/modules/nn_module_v1.py`
    - `/tmpscratch/sliu24/demos/likelihood_data_processed.npz`
- Check if import has the right `nn_module` name
- Run the notebook
- After finished, there will be graphs and the trained model in your working directory.

### Validate an example NN
- Copy the following into the same directory as the `xxxx_model+.pt` file
    - `./archive/v1/validation/Validation.ipynb`
    - All the `likelihood_xxx.npz` files
    - `nn_module_v1.py`
- Run `Validation.ipynb`
- Graphs should be saved to the same directory

<br/>

## Going Beyond Old Code

Above was the state of the project at the end of summer 2021. Since then, a lot has happened, but the basic structure remains the same. Go into the `training` and `validation` folders to learn more about how to execute newer code. `archive` is a self-contained folder with everything needed to reproduce the project at the end of summer 2021. Outside of `archive`, everything is in active development.

Here is a brief overview of each folder:
- `demos`: Minimal runnable code that captures essential ideas
- `models`: Files that contain trained NNs, possibly along with validation graphs
- `modules`: Python modules that need to be imported for every code run in this repository
- `tools`: Handy scripts for a variety of tasks of tangential importance to the project
- `training`: Code for training NNs
- `validation`: Code for validating NNs

Additional Notes:
- Folders named `nb_code` contain the raw code of the Jupyter notebooks in the same directory. This is to keep track of meaningful changes in the notebooks. Therefore, please update the raw code every time a notebook is modified by saving the notebook as a `.py` file.
- Data for the validation and analysis graphs, including the 1D and 2D `combine` scans, are in `CurateND`.

<br/>

## TODOs

- Make sure all validation codes are compatible with the changes associated with compare_plots.
- Use early stopping for training.
- See #TODOs scattered around the repository.
- Link to storage for large files, such as training data and graphing data.