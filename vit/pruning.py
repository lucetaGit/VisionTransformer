import torch
import torch.nn as nn


class Threshold_Pruning(nn.Module):

    def __init__(self, s, c ):
        super(Threshold_Pruning, self).__init__()

        self.s = s
        self.c = c


    def forward(self, x, Th):
        softthreshold1 = x * torch.tanh(self.s*(x-Th))
        softthreshold2 = self.c * torch.tanh(self.s*(x-Th))
        
        x = torch.where(x >= Th, softthreshold1, softthreshold2)
        
        B, H, N, N = x.shape

        sparsity = len(torch.where(x < -self.c/2)[0])/(B*H*N*N)


        return x, sparsity