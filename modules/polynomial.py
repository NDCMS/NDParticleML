"""NN layer that outputs monomials

This layer outputs all possible monomials up to the given degree from
the list of inputs.

"""

import torch
from torch import nn
import math

class MonomialLayer(nn.Module):
    """Outputs all possible monomials up to given degree from inputs.

    The basic idea is to add the number 1 to the list of inputs and
    then create every possible monomial of the given degree from
    factors of the inputs and one.  The presence of one in the list
    generates the monomials with degree less than the given degree.
    See the math of multisets for more information.  (Wikipedia has a
    good entry on this.

    Attributes:
       n_inputs: The number of inputs this layer expects
       degree: The maximum degree of the momomial
       n_outputs: The number of monomials output

    """
    def __init__(self, n_inputs: int, degree: int) -> None:
        """Constructor

        Args:
            n_inputs: The number of input variables to the layer
            degree: The maximum degree of the monomials
        """
        
        super(MonomialLayer,self).__init__()

        self.n_inputs = n_inputs
        self.degree = degree
        # No point in keeping the constant term
        self.n_outputs = int(math.factorial(n_inputs+degree) /
                          math.factorial(n_inputs) /
                          math.factorial(degree)) - 1

        # Now, let's build an array of the indices of the inputs that
        # need to be combined for each monomial
        self.m_ind = torch.zeros(self.n_outputs,degree,dtype=torch.int32).cuda()
        curr_ind = torch.zeros(self.degree, dtype=torch.int32).cuda()

        for row in range(self.n_outputs):
            # Calculate the values for this row
            for col in range(self.degree-1,0,-1):
                if curr_ind[col-1] > curr_ind[col]:
                    curr_ind[col]+=1
                    break
                else:
                    curr_ind[col]=0
            else:
                curr_ind[0]+=1
                curr_ind[1:]=0 # Broadcasts!

            # Set the indices for this row
            self.m_ind[row,:] =  curr_ind

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Performs forward propagation for this module

        Args: 
            x: 
              The input tensor to this layer.  This function preserves
              all indices except the last one, which is assumed to
              index the input variables.  Leading indices can be used
              for minibatches or structuring the inputs.

        """
        
        if x.shape[-1] != self.n_inputs:
            raise IndexError(f'Expecting {self.n_inputs} inputs, got {x.shape[-1]}')
        x = torch.cat((torch.ones(x.shape[:-1]+(1,)).cuda(),x),axis=-1)
        return torch.prod(torch.index_select(x,-1,self.m_ind.flatten())
                          .reshape(x.shape[:-1]+self.m_ind.shape),axis=-1)


class PolynomialLayer(nn.Module):
    """Calculates one or more polynomials of the inputs.

    For a given list of inputs (as few as one) and a given degree,
    calculate one or more polynomials.  The polynomial coefficients
    are trainable weights.

    Attributes:
       n_inputs: The number of inputs this layer expects
       degree: The maximum degree of the polynomial
       n_outputs: The number of polynomials output

    """
    def __init__(self, n_inputs: int,
                 degree: int,
                 n_outputs:int) -> None:
        """Constructor

        Args:
            n_inputs: The number of input variables to the layer
            degree: The maximum degree of the polynomials
        """
        
        super(PolynomialLayer,self).__init__()

        self.n_inputs = n_inputs
        self.degree = degree
        self.n_outputs = n_outputs

        self.monomial_layer = MonomialLayer(self.n_inputs,self.degree)
        self.linear = nn.Linear(self.monomial_layer.n_outputs,self.n_outputs)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Performs forward propagation for this module

        Args: 
            x: 
              The input tensor to this layer.  This function preserves
              all indices except the last one, which is assumed to
              index the input variables.  Leading indices can be used
              for minibatches or structuring the inputs.

        """
        
        if x.shape[-1] != self.n_inputs:
            raise IndexError(f'Expecting {self.n_inputs} inputs, got {x.shape[-1]}')
        return self.linear(self.monomial_layer(x))

class PolySplineLayer(nn.Module):
    """Approximates a function using a combination of multiple polynomials
    controlled by a softmax attention function.

    Attributes:
       n_inputs: The number of inputs this layer expects

    """
    def __init__(self, n_inputs: int,
                 degree: int,
                 n_attn: int,
                 n_outputs:int) -> None:
        """Constructor

        Args:
            n_inputs: The number of input variables to the layer
            degree: The maximum degree of the polynomials
        """
        
        super(PolySplineLayer,self).__init__()

        self.n_inputs = n_inputs
        self.degree = degree
        self.n_attn = n_attn
        self.n_outputs = n_outputs

        self.monomial_layer = MonomialLayer(self.n_inputs,self.degree)
        self.poly_coeffs = nn.ModuleList()
        self.quadratic_layer = MonomialLayer(self.n_inputs,2)
        self.attention = nn.ModuleList()
        for i in range(n_outputs):
            self.poly_coeffs.append(nn.Linear(self.monomial_layer.n_outputs,self.n_attn))
            self.attention.append(nn.Sequential(
                self.quadratic_layer,
                nn.Linear(self.quadratic_layer.n_outputs,self.n_attn),
                nn.ReLU(),
                nn.Softmax(dim=-1),
            ))
                                  
            
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Performs forward propagation for this module

        Args: 
            x: 
              The input tensor to this layer.  This function preserves
              all indices except the last one, which is assumed to
              index the input variables.  Leading indices can be used
              for minibatches or structuring the inputs.

        """
        
        if x.shape[-1] != self.n_inputs:
            raise IndexError(f'Expecting {self.n_inputs} inputs, got {x.shape[-1]}')

        out = [torch.sum(poly(self.monomial_layer(x))*attn(x),dim=-1,keepdim=True)
               for poly,attn in zip(self.poly_coeffs,self.attention)]

        return torch.cat(out)
