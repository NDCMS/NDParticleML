# NDParticleML

This code seeks to construct neural networks which explore dim-6 effective field operators through particle collision data. 

## Overview

The workflow consists of three steps: training, validation, and analysis. Training and validation are completed (although still being improved), but analysis is still to be done.

<br/>

**Training**: During this step, a neural network (NN) is trained on likelihood data from the LHC CMS experiment to approximate the likelihood function (LF) with sufficient accuracy.
- Requirements
    - A `Combine` sampling of the LF
- Products
    - A trained NN as a `pt` file

<br/>

**Validation**: During this step, the trained NN is tested against "correct" data to analyze its accuracy.
- Requirements
    - `Combine` low-dimentional scans of the LF
    - A trained NN
- Products
    - Comparison graphs between `Combine` scans and NN scans

<br/>

**Analysis**: During this step, we take the trained NN as the LF and explore the 16D parameter space, taking advantage of the speedup over `Combine` samplings.
- Requirements
    - A trained NN
- Products
    - TBD

## Contents
