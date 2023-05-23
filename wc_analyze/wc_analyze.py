import nn_module as nnm
import torch
import numpy as np

def Analyze_WC(model_path,**wc):
    #Load the model
    save_dict = torch.load(model_path)
    best_model_state = save_dict['model']
    parameters_save = save_dict['parameters']
    input_stats = save_dict['input_stats']
    output_stats = save_dict['output_stats']
    model = nnm.create_model(26, 1, parameters_save, input_stats, output_stats)
    model.load_state_dict(best_model_state)
    model.eval()
    
    #Establish Avaliable WCs
    names = {}
    names['cQei'] = 0
    names['cQl3i'] = 1
    names['cQlMi'] = 2
    names['cbW'] = 3
    names['cpQ3'] = 4
    names['cpQM'] = 5
    names['cpt'] = 6
    names['cptb'] = 7
    names['ctG'] = 8
    names['ctW'] = 9
    names['ctZ'] = 10
    names['ctei'] = 11
    names['ctlSi'] = 12
    names['ctlTi'] = 13
    names['ctli'] = 14
    names['ctp'] = 15
    names['cQq13'] = 16
    names['cQq83'] = 17
    names['cQq11'] = 18
    names['ctq1'] = 19
    names['cQq81'] = 20
    names['ctq8'] = 21
    names['ctt1'] = 22
    names['cQQ1'] = 23
    names['cQt1'] = 24
    names['cQt8'] = 25

    #Figure out length of the inputs
    try:
        input_length = len(wc[list(wc.keys())[0]])
    except:
        input_length = 1

    inputs = torch.zeros(input_length,26).cuda(0)
    
    #Convert everything to a cuda torch tensor
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
    return (model(inputs) + parameters_save['min'])
