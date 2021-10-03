import torch
import torch.nn as nn

class Adversarial_Loss(nn.Module):
    def __init__(self,nominal_controller,plant):
        super(Adversarial_Loss, self).__init__()
        self.loss = torch.nn.BCELoss()
        self.nominal_controller = nominal_controller
        self.plant = plant
    def forward(self,u_t,x_t,v_t,discriminator,label):
        u_t1 = self.nominal_controller(x_t,v_t) + u_t
        delta_x = self.plant(x_t,u_t1)
        fake = torch.cat((v_t, delta_x), 1)
        output = discriminator(fake)
        return self.loss(output,label)

class Evaluate_Loss(nn.Module):
    def __init__(self,nominal_controller,plant):
        super(Evaluate_Loss, self).__init__()
        self.loss = torch.nn.MSELoss()
        self.nominal_controller = nominal_controller
        self.plant = plant
    def forward(self,generator,x_t,v_t):
        b_in = torch.cat((v_t, x_t), 1)
        u_out = generator(b_in)
        u_out1 = self.nominal_controller(x_t, v_t) + u_out
        delta_x = self.plant(x_t, u_out1)
        return self.loss(delta_x,v_t)

class Evaluate_Model(nn.Module):
    def __init__(self,nominal_controller,plant):
        super(Evaluate_Model, self).__init__()
        self.loss = torch.nn.MSELoss()
        self.model = plant
        self.nominal_controller = nominal_controller
    def forward(self, generator,x_t,v_t):
        beta = -1.0 * (self.model.Plant_F(x_t)/self.model.Plant_G(x_t))
        alpha = 1.0 / self.model.Plant_G(x_t)
        control_plant = beta + alpha * v_t
        b_in = torch.cat((v_t, x_t), 1)
        u_out = generator(b_in)
        control_nominal = self.nominal_controller(x_t, v_t) + u_out
        return self.loss(control_plant,control_nominal)