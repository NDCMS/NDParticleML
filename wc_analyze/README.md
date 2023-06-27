# Wilson Coefficient Analysis

This folder contains everything needed to analyze the dNLL of Wilson Coefficient values.
- `wc_analyze.ipynb`: A notebook to easily analyze the dNLLs of different WCs.
- 'wc_analyze.py': A script that holds the analysis function.

## Running the Code
To analyze a given Wilson Coefficient, either use the jupyter notebook or the python script.
The function that gives the dNLL of certain WCs is called `Analyze_WC`.

def Analyze_WC(model_path='./16603_11_model+.pt',**wc)

The function first takes in the model path, then it takes in the Wilson coefficient values. The Wilson coefficients can either be put in as a single value, a list, a numpy array, or a torch tensor. It returns the dNLL of the given set of WCs.

### Important Note 
The data the DNN was trained had a minimum of around -42. Since it is better for a DNN to be trained on data that has a minimum of 0, the neural network was trained with -42 subtracted from its outputs. Thus, this function adds the original minimum back into the output, so, if combine outputted -42, this function also outputs that value (whereas, on its own, the DNN would output 0).

### Example

a = 1
b = 4
c = 7

Analyze_WC(model_path='./16603_11_model+.pt', cQei = a, cQl3i = b, cQlMi = c)

OR

a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
c = torch.tensor([7,8,9])

Analyze_WC(model_path='./16603_11_model+.pt', cQei = a, cQl3i = b, cQlMi = c)
