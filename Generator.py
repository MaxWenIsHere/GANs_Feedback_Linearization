import torch
import torch.nn as nn
class Generator_MLP(nn.Module):
    def __init__(self,n_feature, size_hidden):
        super(Generator_MLP, self).__init__()
        self.hidden1 = nn.Sequential(nn.Linear(n_feature, size_hidden),nn.LeakyReLU(0.2))
        self.hidden2 = nn.Sequential(nn.Linear(size_hidden, size_hidden),nn.LeakyReLU(0.2))
        self.predict = torch.nn.Linear(size_hidden, 1)
    def forward(self, x):
        x1 = self.hidden1(x)
        x2 = self.hidden2(x1)
        y = self.predict(x2)
        return y
class Generator_LSTM(nn.Module):
    def __init__(self,n_feature, size_hidden):
        super(Generator_LSTM, self).__init__()
        self.hidden1 = nn.Sequential(nn.Linear(n_feature, size_hidden),nn.LeakyReLU(0.2))
        self.LSTM = nn.LSTM(input_size=1,hidden_size=1,num_layers=1,bidirectional=False,batch_first=False)
        self.predict = torch.nn.Linear(size_hidden, 1)
    def forward(self, x):
        embedding = self.hidden1(x)
        embedding = embedding.view(len(embedding), 1, -1)
        embedding = embedding.permute(2,0,1)
        r_out, (h_n, h_c) = self.LSTM(embedding)
        r_out = r_out.squeeze(2)
        r_out = r_out.permute(1, 0)
        y = self.predict(r_out)
        return y
class Generator_CNN(nn.Module):
    def __init__(self,n_feature, size_hidden,out_channel,kernel_size):
        super(Generator_CNN, self).__init__()
        self.hidden1 = nn.Sequential(nn.Linear(n_feature, size_hidden),nn.LeakyReLU(0.2))
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_channel, kernel_size=kernel_size)
        self.predict = torch.nn.Linear((size_hidden-kernel_size+1)*out_channel, 1)
    def forward(self, x):
        x1 = self.hidden1(x)
        x1 = x1.view(len(x1), 1, -1)
        x2 = self.conv1(x1)
        x2 = x2.view(len(x2), -1)
        y = self.predict(x2)
        return y