# Wilson Coefficient Analysis

This folder contains everything needed to analyze the dNLL of Wilson Coefficient (WC) values.
- `wc_analyze.ipynb`: A notebook to easily analyze the dNLLs of different WCs.
- 'wc_analyze.py': A script that holds the analysis function.

## Running the Code
To analyze a given set of WCs, use either the Jupyter Notebook or the Python script.
The function responsible for calculating the dNLL for a particular set of WCs is referred to as `Analyze_WC`.

def Analyze_WC(model_path='./16603_11_model+.pt',**wc)

The function accepts the model path as the first argument, followed by the WC values. These values can be provided as a single value, a list, a NumPy array, or a Torch tensor. The function calculates and returns the dNLL for the inputted WCs.

### Important Note 
The training data used for the DNN had a minimum of around -42. To ensure that the DNN was trained with a minimum of 0, the network was trained by subtracting -42 from the target outputs. Therefore, this function adds -42 into its output in order to restore the original minimum value.

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
