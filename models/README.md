# Models

This folder contains good trained models.

`6755_13` is the best model until October 2022. Hyperparameters used:
- Training file: `likelihood_data_no_delta.npz`
- Training size: 49378587
- Validation size: 498773
- Nodes per hidden layer: 750
- Hidden layers: 2
- Minibatch size: 1024
- Epochs: 1000
- Activation function: ReLU
- Loss function: MSE
- Optimizer: `ADAM`
- Initial learning rate: 10<sup>-4</sup>
- Learning Rate Reduction Factor: 0.2
- Learning Rate Reduction Patience: 20
- Learning Rate Reduction Threshold: 10<sup>-6</sup>
- Weight Decay: 0.000000

`12643_0` is the best model so far. Hyperparameters used:
- Training file: `likelihood_data_no_delta.npz`
- Training size: 49378587
- Validation size: 498773
- Nodes per hidden layer: 700
- Hidden layers: 2
- Minibatch size: 512
- Epochs: 500
- Activation function: ReLU
- Loss function: MSE
- Optimizer: `ADAM`
- Initial learning rate: 10<sup>-4</sup>
- Learning Rate Reduction Factor: 0.2
- Learning Rate Reduction Patience: 20
- Learning Rate Reduction Threshold: 10<sup>-6</sup>
- Weight Decay: 0.000000