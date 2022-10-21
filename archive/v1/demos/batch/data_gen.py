import torch

# Target function to approximate
def target_func(input):
    """
    This is the fucntion the network will try to approximate.

    Inputs: input (Pytorch tensor)

    Outputs: output (Pytorch tensor)
    """
    return torch.unsqueeze(input[:,0]**2 + input[:,1], 1)

# Notebook

# Hyperparameters
parameters = {}
parameters['N'] = 2
parameters['train_size'] = 1000000
parameters['test_size'] = 10000

# Create a list of sets of N random numbers, the training set
inputs = (10 * torch.rand((parameters['train_size'],parameters['N']),dtype=torch.float32) - 5)
outputs = target_func(inputs)

# create a test set
test_inputs = (10 * torch.rand((parameters['test_size'],parameters['N']),dtype=torch.float32) - 5) # Uniform
# test_inputs = torch.randn((test_size,N),dtype=torch.float32) # Normal
test_outputs = target_func(test_inputs)

torch.save(inputs, 'inputs.pt', _use_new_zipfile_serialization=False)
torch.save(outputs, 'outputs.pt', _use_new_zipfile_serialization=False)
torch.save(test_inputs, 'test_inputs.pt', _use_new_zipfile_serialization=False)
torch.save(test_outputs, 'test_outputs.pt', _use_new_zipfile_serialization=False)