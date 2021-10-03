import torch
import torch.nn as nn
import numpy as np

class Plant_Sys(nn.Module):
    def __init__(self):
        super(Plant_Sys, self).__init__()
    def Plant_F(self,state):
        return 1.0 - torch.sin(state) + (0.5 / (1.0 + torch.exp(-state/10.0)))
    def Plant_G(self,state):
        return 1.0 + 0.5 * torch.sin(state*0.5)
    def forward(self, state,input):
        delta_x = self.Plant_F(state) + self.Plant_G(state) * input
        return delta_x

class Nominal_Control(nn.Module):
    def __init__(self,factor_f,factor_g,model):
        super(Nominal_Control, self).__init__()
        self.factor_f = factor_f
        self.factor_g = factor_g
        self.model = model
    def Nominal_F(self, state):
        return self.model.Plant_F(state) * self.factor_f
    def Nominal_G(self, state):
        if np.abs(self.factor_g) < 0.01:
            return float('inf')
        else:
            return self.model.Plant_G(state) * self.factor_g
    def forward(self, state,input_virtual):
        beta = -1.0 * (self.Nominal_F(state)/self.Nominal_G(state))
        alpha = 1.0 / self.Nominal_G(state)
        control_u = beta + alpha * input_virtual
        return control_u

