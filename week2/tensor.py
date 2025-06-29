# -*- coding: utf-8 -*-
"""tensor_ops.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10C57jgW8aeJwSzCU_IS62L70L7E_FM9f
"""

import torch

my_torch=torch.arange(10)
my_torch

#reshape and view
my_torch = my_torch.reshape(2,5)
my_torch

my_torch2 = torch.arange(15)
my_torch2

my_torch2=my_torch2.reshape(3,-1)
my_torch2

#with reshape and view , they will update
my_torch5 = torch.arange(10)
my_torch5

my_torch6=my_torch5.reshape(2,5)
my_torch6

my_torch5[1]=4141
my_torch5

my_torch7=torch.arange(10)
my_torch7

my_torch8=torch.arange(10)
my_torch8

my_torch8=my_torch7.reshape(5,2)
my_torch8

my_torch8[:,0]

my_torch8[:,1:]

my_torch8[0,:]

# -*- coding: utf-8 -*-
"""Tensor_Math.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1W-hKqops0pCpkZiJiaSWYAds_S0MfpAi

tensor math
"""

import torch
import numpy as np

tensor_a = torch.tensor([1,2,3,4])
tensor_b= torch.tensor([5,6,7,8])

#addition
tensor_a+tensor_b

# addition longhand
torch.add(tensor_a,tensor_b)

# subtraction
tensor_a-tensor_b

#remainder
tensor_a%tensor_b

#exponent
torch.pow(tensor_a,tensor_b)

#exponent
tensor_a**tensor_b

#tensor_a=tensor_a+tensor_b
tensor_a.add_(tensor_b)
