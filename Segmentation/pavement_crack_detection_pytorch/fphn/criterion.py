import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        side_network_1, side_network_2, side_network_3, side_network_4, side_network_5, final_outputs = targets
        loss_1 = nn.CrossEntropyLoss()(outputs, side_network_1)
        loss_2 = nn.CrossEntropyLoss()(outputs, side_network_2)
        loss_3 = nn.CrossEntropyLoss()(outputs, side_network_3)
        loss_4 = nn.CrossEntropyLoss()(outputs, side_network_4)
        loss_5 = nn.CrossEntropyLoss()(outputs, side_network_5)
        loss_lass = nn.CrossEntropyLoss()(outputs, final_outputs)
        total_loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5
        return total_loss
