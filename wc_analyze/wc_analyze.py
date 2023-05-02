import nn_module as nnm
import torch
import numpy as np

def Analyze_WC(model_path,**wc):
    #Load the model
    save_dict = torch.load('model_path')
    best_model_state = save_dict['model']
    parameters_save = save_dict['parameters']
    input_stats = save_dict['input_stats']
    output_stats = save_dict['output_stats']
    model = nnm.create_model(26, 1, parameters_save)
    model.load_state_dict(best_model_state)
    model.eval()

    try:
        input_length = len(wc[list(wc.keys())[0]])
    except:
        input_length = 1

    inputs = torch.zeros(input_length,26).cuda(0)
    
    for key in wc:
        try:
            if(type(wc[key]) == torch.Tensor):
                inputs[:,names[key]] = wc[key].cuda(0)
            elif(type(wc[key]) == np.ndarray):
                inputs[:,names[key]] = torch.from_numpy(wc[key]).cuda(0)
            else:
                inputs[:,names[key]] = torch.tensor([wc[key]]).cuda(0)
        except:
            raise RuntimeError(f'The Wilson Coefficient {key} is not supported.')
    print(inputs)
    return model(inputs)
