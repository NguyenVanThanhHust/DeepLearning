import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

# Assume input is (4,1) vector
class BasicRNN(nn.Module):
    def __init__(self, n_inputs, n_neurons):
        super(BasicRNN, self).__init__()

        self.Wx = torch.randn(n_inputs, n_neurons) # 4x1
        self.Wy = torch.randn(n_neurons, n_neurons) # 1x1
        self.b = torch.zeros(1, n_neurons) # 1x4

    def forward(self, X0, X1):
        self.Y0 = torch.tanh(torch.mm(X0, self.Wx) + self.b) # 4x1
        self.Y1 = torch.tanh(torch.mm(self.Y0, self.Wy) + 
                            torch.mm(X1, self.Wx) + self.b)
        return self.Y0, self.Y1

N_INPUT = 3
N_NEURONS = 5

X0_batch = torch.tensor([[0,1,2], [3,4,5], 
                         [6,7,8], [9,0,1]],
                        dtype = torch.float) #t=0 => 4 X 3

X1_batch = torch.tensor([[9,8,7], [0,0,0], 
                         [6,5,4], [3,2,1]],
                        dtype = torch.float) #t=1 => 4 X 3

model = BasicRNN(N_INPUT, N_NEURONS)

Y0_val, Y1_val = model(X0_batch, X1_batch)
print("Y0_val: ", Y0_val)
print("Y1_val: ", Y1_val)
