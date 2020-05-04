import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class CleanBasicRNN(nn.Module):
    def __init__(self, batch_size, n_inputs, n_neurons):
        super(CleanBasicRNN, self).__init__()
        self.rnn = nn.RNNCell(n_inputs, n_neurons)
        self.hidden_state = torch.randn(batch_size, n_neurons) # initialize hidden state

    def forward(self, X):
        output = []

        # for each image step
        for i in range(2):
            self.hidden_state = self.rnn(X[i], self.hidden_state)
            output.append(self.hidden_state)

        return output, self.hidden_state

FIXED_BATCH_SIZE = 4 # our batch size is fixed for now
N_INPUT = 3
N_NEURONS = 5

X_batch = torch.tensor([[[0,1,2], [3,4,5], 
                         [6,7,8], [9,0,1]],
                        [[9,8,7], [0,0,0], 
                         [6,5,4], [3,2,1]]
                       ], dtype = torch.float) # X0 and X1

model = CleanBasicRNN(FIXED_BATCH_SIZE, N_INPUT, N_NEURONS)

output_val, states_val = model(X_batch)
print(output_val) # contains all output for all timesteps
print(states_val) # contains values for final state or final timestep, i.e., t=1
