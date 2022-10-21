# Models

This folder contains the best model from summer 2021. Hyperparameters used:
- Training file: `likelihood_data_processed.npz`
- Training size: ~7M
- Validation size: ~70K
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