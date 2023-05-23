# Wilson Coefficient Analysis

This folder contatins everything needed to analyze the dNLL of Wilson Coefficient values.
- `wc_analyze.ipynb`: A notebook to easily analyze different WCs.
- 'wc_analyze.py': A script that holds the analysis function.

## Running the Code
To analyze a given Wilson Coefficient, either use the jupyter notebook or the python script.
The function that gives the dNLL of certain WC parameters is called `Analyze_WC`.

def Analyze_WC(model_path='./16603_11_model+.pt',**wc)

The function first takes in the model path, then it takes in the wilson coefficient values. The wilson coefficients can either be put in as a single value, a list, a numpy array, or a torch tensor. It returns the dNLL of the given set of WCs.

An imporant note. The data the DNN was trained had a minimum of around -42. Thus, since the it is better for a DNN to be trained on data that has a minimum of 0, the original data from combine was convereted to this minimum of zero. This function adds the original minimum back into the output, so, if combine outputeed -42, this function also would also output that value (whereas, on its own, the DNN would output 0).

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
