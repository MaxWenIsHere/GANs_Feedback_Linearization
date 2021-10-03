import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as Data
import scipy.io as sio
from torch.autograd import Variable
from scipy.io import savemat


torch.manual_seed(20210914)
np.random.seed(20210914)


data = sio.loadmat('integral_data.mat')
data = data['save_data']
ss = MinMaxScaler()
#data = ss.fit_transform(data)

plt.plot(data[:,0])
plt.plot(data[:,1])
plt.show()


v_train = data[:,0].reshape(-1,1)
x_train = data[:,1].reshape(-1,1)
v_train = torch.Tensor(v_train)
x_train = torch.Tensor(x_train)
x_train, v_train = Variable(x_train), Variable(v_train)
torch_dataset = Data.TensorDataset(v_train, x_train)
loader = Data.DataLoader(dataset=torch_dataset,batch_size=1024,shuffle=True)



class Net(torch.nn.Module):
    def __init__(self, n_feature, size_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Sequential(nn.Linear(n_feature, size_hidden), nn.LeakyReLU(0.2))
        self.hidden2 = nn.Sequential(nn.Linear(size_hidden, size_hidden), nn.LeakyReLU(0.2))
        self.predict = torch.nn.Linear(size_hidden, n_output)

    def forward(self, x):
        x1 = self.hidden1(x)
        x2 = self.hidden2(x1)
        y = self.predict(x2)
        return y

class loss_define(torch.nn.Module):
    def __init__(self):
        super(loss_define, self).__init__()
        self.loss = torch.nn.MSELoss()
    def forward(self,u_t,x_t,v_t):
        delta_x = 1.0 - torch.sin(x_t) + (0.5 / (1.0 + torch.exp(-x_t/10.0))) + (1.0 + 0.5 * torch.sin(x_t*0.5)) * u_t
        return self.loss(delta_x,v_t)

def initialize_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)

net = Net(2, 10, 1)
net.apply(initialize_weight)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
criterion = loss_define()

EPOCH = 2000
for epoch in range(EPOCH):
    loss_ = 0
    for step, (batch_v, batch_x) in enumerate(loader):  # for each training step
        b_x = Variable(batch_x)
        b_v = Variable(batch_v)
        b_in = torch.cat((b_v,b_x),1)
        u_out = net(b_in)  # input x and predict based on x
        loss = criterion(u_out, b_x,b_v)  # must be (1. nn output, 2. target)
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()
        loss_ += loss.item()
    print(epoch,loss_/step)

hidden1_w = net.hidden1[0].weight.data.numpy()
hidden1_b = net.hidden1[0].bias.data.numpy()
hidden2_w = net.hidden2[0].weight.data.numpy()
hidden2_b = net.hidden2[0].bias.data.numpy()
out_w = net.predict.weight.data.numpy()
out_b = net.predict.bias.data.numpy()
savemat('model_Regression.mat', {'hidden1_w':hidden1_w,'hidden1_b':hidden1_b,'hidden2_w':hidden2_w,'hidden2_b':hidden2_b,'out_w':out_w,'out_b':out_b})