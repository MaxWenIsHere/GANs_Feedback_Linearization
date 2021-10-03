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
from Model import Plant_Sys,Nominal_Control
from Generator import Generator_MLP,Generator_CNN,Generator_LSTM
from Discriminator import Discriminator_MLP,Discriminator_CNN,Discriminator_LSTM
from Evaluation import Adversarial_Loss,Evaluate_Loss,Evaluate_Model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(20210914)
np.random.seed(20210914)

#
# torch.manual_seed(20210915)
# np.random.seed(20210915)

# data = sio.loadmat('integral_data.mat')
# data = data['save_data']
# ss = MinMaxScaler(feature_range=(-1, 1))
#data = ss.fit_transform(data)

data = (10.0 - (-10.0)) * np.random.random((20000,2)) + (-10.0)
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

factor_f = 1.5
factor_g = 1.5


plant = Plant_Sys()
nominal_controller = Nominal_Control(factor_f,factor_g,plant)




def initialize_weight(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)
    # elif isinstance(m, nn.LSTM):
    #     nn.init.xavier_normal_(m.weight.data, gain = nn.init.calculate_gain('relu'))
    #     nn.init.constant_(m.bias.data, 0)

EPOCH = 2000
print("1 GMDM")
##1 GMDM
criterion = torch.nn.BCELoss()
adversarial_loss = Adversarial_Loss(nominal_controller,plant)
evaluate_loss = Evaluate_Loss(nominal_controller,plant)
evaluate_model = Evaluate_Model(nominal_controller,plant)
hidden_dim = 10
CNN_Out_Channel = 2
CNN_Kernel_Size = 4
generator = Generator_MLP(2,hidden_dim)#,CNN_Out_Channel,CNN_Kernel_Size)
discriminator_ = Discriminator_MLP(2,hidden_dim)#,CNN_Out_Channel,CNN_Kernel_Size)
generator.apply(initialize_weight)
discriminator_.apply(initialize_weight)
d_optimizer = torch.optim.Adam(discriminator_.parameters(), lr=0.0001)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
data_save1 = []
data_save2 = []
for epoch in range(EPOCH):
    evaluate_loss_ = 0
    evaluate_mode_ = 0
    for step, (batch_v, batch_x) in enumerate(loader):  # for each training step
        b_x = Variable(batch_x).to(device)
        b_v = Variable(batch_v).to(device)
        b_in = torch.cat((b_v,b_x),1)
        real = torch.cat((b_v,b_v),1)
        num_batch = batch_v.size(0)
        real_label = Variable(torch.ones([num_batch,1]))
        fake_label = Variable(torch.zeros([num_batch,1]))

        d_optimizer.zero_grad()
        output = discriminator_(real)
        errD_real = criterion(output, real_label)
        errD_real.backward()

        u_out = generator(b_in)
        errD_fake = adversarial_loss(u_out,b_x,b_v,discriminator_,fake_label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        d_optimizer.step()

        g_optimizer.zero_grad()
        u_out = generator(b_in)
        errG = adversarial_loss(u_out, b_x, b_v, discriminator_, real_label)
        errG.backward()
        g_optimizer.step()

        evaluate_loss_ += evaluate_loss(generator,b_x,b_v)
        evaluate_mode_ += evaluate_model(generator, b_x, b_v)

    print(epoch,evaluate_loss_.item()/step,evaluate_mode_.item()/step)
    data_save1.append(evaluate_loss_.item()/step)
    data_save2.append(evaluate_mode_.item()/step)

data_save1 = np.asarray(data_save1)
data_save2 = np.asarray(data_save2)
savemat('GMDM_loss.mat', {'GMDM_loss':data_save1})
savemat('GMDM_u.mat', {'GMDM_u':data_save2})

hidden1_w = generator.hidden1[0].weight.data.numpy()
hidden1_b = generator.hidden1[0].bias.data.numpy()
hidden2_w = generator.hidden2[0].weight.data.numpy()
hidden2_b = generator.hidden2[0].bias.data.numpy()
out_w = generator.predict.weight.data.numpy()
out_b = generator.predict.bias.data.numpy()
savemat('model_GAN.mat', {'factor_f':factor_f,'factor_g':factor_g,'hidden1_w':hidden1_w,'hidden1_b':hidden1_b,'hidden2_w':hidden2_w,'hidden2_b':hidden2_b,'out_w':out_w,'out_b':out_b})




# #2 GMDC
# print("2 GMDC")
# criterion = torch.nn.BCELoss()
# adversarial_loss = Adversarial_Loss(nominal_controller,plant)
# evaluate_loss = Evaluate_Loss(nominal_controller,plant)
# evaluate_model = Evaluate_Model(nominal_controller,plant)
# hidden_dim = 10
# CNN_Out_Channel = 2
# CNN_Kernel_Size = 4
# generator = Generator_MLP(2,hidden_dim)#,CNN_Out_Channel,CNN_Kernel_Size)
# discriminator_ = Discriminator_CNN(2,hidden_dim,CNN_Out_Channel,CNN_Kernel_Size)
# generator.apply(initialize_weight)
# discriminator_.apply(initialize_weight)
# d_optimizer = torch.optim.Adam(discriminator_.parameters(), lr=0.0001)
# g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
# data_save1 = []
# data_save2 = []
# for epoch in range(EPOCH):
#     evaluate_loss_ = 0
#     evaluate_mode_ = 0
#     for step, (batch_v, batch_x) in enumerate(loader):  # for each training step
#         b_x = Variable(batch_x).to(device)
#         b_v = Variable(batch_v).to(device)
#         b_in = torch.cat((b_v,b_x),1)
#         real = torch.cat((b_v,b_v),1)
#         num_batch = batch_v.size(0)
#         real_label = Variable(torch.ones([num_batch,1]))
#         fake_label = Variable(torch.zeros([num_batch,1]))
#
#         d_optimizer.zero_grad()
#         output = discriminator_(real)
#         errD_real = criterion(output, real_label)
#         errD_real.backward()
#
#         u_out = generator(b_in)
#         errD_fake = adversarial_loss(u_out,b_x,b_v,discriminator_,fake_label)
#         errD_fake.backward()
#         errD = errD_real + errD_fake
#         d_optimizer.step()
#
#         g_optimizer.zero_grad()
#         u_out = generator(b_in)
#         errG = adversarial_loss(u_out, b_x, b_v, discriminator_, real_label)
#         errG.backward()
#         g_optimizer.step()
#
#         evaluate_loss_ += evaluate_loss(generator,b_x,b_v)
#         evaluate_mode_ += evaluate_model(generator, b_x, b_v)
#
#     print(epoch,evaluate_loss_.item()/step,evaluate_mode_.item()/step)
#     data_save1.append(evaluate_loss_.item()/step)
#     data_save2.append(evaluate_mode_.item()/step)
#
# data_save1 = np.asarray(data_save1)
# data_save2 = np.asarray(data_save2)
# savemat('GMDC_loss.mat', {'GMDC_loss':data_save1})
# savemat('GMDC_u.mat', {'GMDC_u':data_save2})
# #
# #
# ##3 GMDL
# print("3 GMDL")
# criterion = torch.nn.BCELoss()
# adversarial_loss = Adversarial_Loss(nominal_controller,plant)
# evaluate_loss = Evaluate_Loss(nominal_controller,plant)
# evaluate_model = Evaluate_Model(nominal_controller,plant)
# hidden_dim = 10
# CNN_Out_Channel = 2
# CNN_Kernel_Size = 4
# generator = Generator_MLP(2,hidden_dim)#,CNN_Out_Channel,CNN_Kernel_Size)
# discriminator_ = Discriminator_LSTM(2,hidden_dim)#,CNN_Out_Channel,CNN_Kernel_Size)
# generator.apply(initialize_weight)
# discriminator_.apply(initialize_weight)
# d_optimizer = torch.optim.Adam(discriminator_.parameters(), lr=0.0001)
# g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
# data_save1 = []
# data_save2 = []
# for epoch in range(EPOCH):
#     evaluate_loss_ = 0
#     evaluate_mode_ = 0
#     for step, (batch_v, batch_x) in enumerate(loader):  # for each training step
#         b_x = Variable(batch_x).to(device)
#         b_v = Variable(batch_v).to(device)
#         b_in = torch.cat((b_v,b_x),1)
#         real = torch.cat((b_v,b_v),1)
#         num_batch = batch_v.size(0)
#         real_label = Variable(torch.ones([num_batch,1]))
#         fake_label = Variable(torch.zeros([num_batch,1]))
#
#         d_optimizer.zero_grad()
#         output = discriminator_(real)
#         errD_real = criterion(output, real_label)
#         errD_real.backward()
#
#         u_out = generator(b_in)
#         errD_fake = adversarial_loss(u_out,b_x,b_v,discriminator_,fake_label)
#         errD_fake.backward()
#         errD = errD_real + errD_fake
#         d_optimizer.step()
#
#         g_optimizer.zero_grad()
#         u_out = generator(b_in)
#         errG = adversarial_loss(u_out, b_x, b_v, discriminator_, real_label)
#         errG.backward()
#         g_optimizer.step()
#
#         evaluate_loss_ += evaluate_loss(generator,b_x,b_v)
#         evaluate_mode_ += evaluate_model(generator, b_x, b_v)
#
#     print(epoch,evaluate_loss_.item()/step,evaluate_mode_.item()/step)
#     data_save1.append(evaluate_loss_.item()/step)
#     data_save2.append(evaluate_mode_.item()/step)
#
# data_save1 = np.asarray(data_save1)
# data_save2 = np.asarray(data_save2)
# savemat('GMDL_loss.mat', {'GMDL_loss':data_save1})
# savemat('GMDL_u.mat', {'GMDL_u':data_save2})
# #
# #
# ##4 GCDM
# print("4 GCDM")
# criterion = torch.nn.BCELoss()
# adversarial_loss = Adversarial_Loss(nominal_controller,plant)
# evaluate_loss = Evaluate_Loss(nominal_controller,plant)
# evaluate_model = Evaluate_Model(nominal_controller,plant)
# hidden_dim = 10
# CNN_Out_Channel = 2
# CNN_Kernel_Size = 4
# generator = Generator_CNN(2,hidden_dim,CNN_Out_Channel,CNN_Kernel_Size)
# discriminator_ = Discriminator_MLP(2,hidden_dim)#,CNN_Out_Channel,CNN_Kernel_Size)
# generator.apply(initialize_weight)
# discriminator_.apply(initialize_weight)
# d_optimizer = torch.optim.Adam(discriminator_.parameters(), lr=0.0001)
# g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
# data_save1 = []
# data_save2 = []
# for epoch in range(EPOCH):
#     evaluate_loss_ = 0
#     evaluate_mode_ = 0
#     for step, (batch_v, batch_x) in enumerate(loader):  # for each training step
#         b_x = Variable(batch_x).to(device)
#         b_v = Variable(batch_v).to(device)
#         b_in = torch.cat((b_v,b_x),1)
#         real = torch.cat((b_v,b_v),1)
#         num_batch = batch_v.size(0)
#         real_label = Variable(torch.ones([num_batch,1]))
#         fake_label = Variable(torch.zeros([num_batch,1]))
#
#         d_optimizer.zero_grad()
#         output = discriminator_(real)
#         errD_real = criterion(output, real_label)
#         errD_real.backward()
#
#         u_out = generator(b_in)
#         errD_fake = adversarial_loss(u_out,b_x,b_v,discriminator_,fake_label)
#         errD_fake.backward()
#         errD = errD_real + errD_fake
#         d_optimizer.step()
#
#         g_optimizer.zero_grad()
#         u_out = generator(b_in)
#         errG = adversarial_loss(u_out, b_x, b_v, discriminator_, real_label)
#         errG.backward()
#         g_optimizer.step()
#
#         evaluate_loss_ += evaluate_loss(generator,b_x,b_v)
#         evaluate_mode_ += evaluate_model(generator, b_x, b_v)
#
#     print(epoch,evaluate_loss_.item()/step,evaluate_mode_.item()/step)
#     data_save1.append(evaluate_loss_.item()/step)
#     data_save2.append(evaluate_mode_.item()/step)
#
# data_save1 = np.asarray(data_save1)
# data_save2 = np.asarray(data_save2)
# savemat('GCDM_loss.mat', {'GCDM_loss':data_save1})
# savemat('GCDM_u.mat', {'GCDM_u':data_save2})
# #
# #
# ##5 GLDM
# print("5 GLDM")
# criterion = torch.nn.BCELoss()
# adversarial_loss = Adversarial_Loss(nominal_controller,plant)
# evaluate_loss = Evaluate_Loss(nominal_controller,plant)
# evaluate_model = Evaluate_Model(nominal_controller,plant)
# hidden_dim = 10
# CNN_Out_Channel = 2
# CNN_Kernel_Size = 4
# generator = Generator_LSTM(2,hidden_dim)#,CNN_Out_Channel,CNN_Kernel_Size)
# discriminator_ = Discriminator_MLP(2,hidden_dim)#,CNN_Out_Channel,CNN_Kernel_Size)
# generator.apply(initialize_weight)
# discriminator_.apply(initialize_weight)
# d_optimizer = torch.optim.Adam(discriminator_.parameters(), lr=0.0001)
# g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
# data_save1 = []
# data_save2 = []
# for epoch in range(EPOCH):
#     evaluate_loss_ = 0
#     evaluate_mode_ = 0
#     for step, (batch_v, batch_x) in enumerate(loader):  # for each training step
#         b_x = Variable(batch_x).to(device)
#         b_v = Variable(batch_v).to(device)
#         b_in = torch.cat((b_v,b_x),1)
#         real = torch.cat((b_v,b_v),1)
#         num_batch = batch_v.size(0)
#         real_label = Variable(torch.ones([num_batch,1]))
#         fake_label = Variable(torch.zeros([num_batch,1]))
#
#         d_optimizer.zero_grad()
#         output = discriminator_(real)
#         errD_real = criterion(output, real_label)
#         errD_real.backward()
#
#         u_out = generator(b_in)
#         errD_fake = adversarial_loss(u_out,b_x,b_v,discriminator_,fake_label)
#         errD_fake.backward()
#         errD = errD_real + errD_fake
#         d_optimizer.step()
#
#         g_optimizer.zero_grad()
#         u_out = generator(b_in)
#         errG = adversarial_loss(u_out, b_x, b_v, discriminator_, real_label)
#         errG.backward()
#         g_optimizer.step()
#
#         evaluate_loss_ += evaluate_loss(generator,b_x,b_v)
#         evaluate_mode_ += evaluate_model(generator, b_x, b_v)
#
#     print(epoch,evaluate_loss_.item()/step,evaluate_mode_.item()/step)
#     data_save1.append(evaluate_loss_.item()/step)
#     data_save2.append(evaluate_mode_.item()/step)
#
# data_save1 = np.asarray(data_save1)
# data_save2 = np.asarray(data_save2)
# savemat('GLDM_loss.mat', {'GLDM_loss':data_save1})
# savemat('GLDM_u.mat', {'GLDM_u':data_save2})
# #
# #
# ##6 GCDC
# print("6 GCDC")
# criterion = torch.nn.BCELoss()
# adversarial_loss = Adversarial_Loss(nominal_controller,plant)
# evaluate_loss = Evaluate_Loss(nominal_controller,plant)
# evaluate_model = Evaluate_Model(nominal_controller,plant)
# hidden_dim = 10
# CNN_Out_Channel = 2
# CNN_Kernel_Size = 4
# generator = Generator_CNN(2,hidden_dim,CNN_Out_Channel,CNN_Kernel_Size)
# discriminator_ = Discriminator_CNN(2,hidden_dim,CNN_Out_Channel,CNN_Kernel_Size)
# generator.apply(initialize_weight)
# discriminator_.apply(initialize_weight)
# d_optimizer = torch.optim.Adam(discriminator_.parameters(), lr=0.0001)
# g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
# data_save1 = []
# data_save2 = []
# for epoch in range(EPOCH):
#     evaluate_loss_ = 0
#     evaluate_mode_ = 0
#     for step, (batch_v, batch_x) in enumerate(loader):  # for each training step
#         b_x = Variable(batch_x).to(device)
#         b_v = Variable(batch_v).to(device)
#         b_in = torch.cat((b_v,b_x),1)
#         real = torch.cat((b_v,b_v),1)
#         num_batch = batch_v.size(0)
#         real_label = Variable(torch.ones([num_batch,1]))
#         fake_label = Variable(torch.zeros([num_batch,1]))
#
#         d_optimizer.zero_grad()
#         output = discriminator_(real)
#         errD_real = criterion(output, real_label)
#         errD_real.backward()
#
#         u_out = generator(b_in)
#         errD_fake = adversarial_loss(u_out,b_x,b_v,discriminator_,fake_label)
#         errD_fake.backward()
#         errD = errD_real + errD_fake
#         d_optimizer.step()
#
#         g_optimizer.zero_grad()
#         u_out = generator(b_in)
#         errG = adversarial_loss(u_out, b_x, b_v, discriminator_, real_label)
#         errG.backward()
#         g_optimizer.step()
#
#         evaluate_loss_ += evaluate_loss(generator,b_x,b_v)
#         evaluate_mode_ += evaluate_model(generator, b_x, b_v)
#
#     print(epoch,evaluate_loss_.item()/step,evaluate_mode_.item()/step)
#     data_save1.append(evaluate_loss_.item()/step)
#     data_save2.append(evaluate_mode_.item()/step)
#
# data_save1 = np.asarray(data_save1)
# data_save2 = np.asarray(data_save2)
# savemat('GCDC_loss.mat', {'GCDC_loss':data_save1})
# savemat('GCDC_u.mat', {'GCDC_u':data_save2})
# #
# #
# ##7 GCDL
# print("7 GCDL")
# criterion = torch.nn.BCELoss()
# adversarial_loss = Adversarial_Loss(nominal_controller,plant)
# evaluate_loss = Evaluate_Loss(nominal_controller,plant)
# evaluate_model = Evaluate_Model(nominal_controller,plant)
# hidden_dim = 10
# CNN_Out_Channel = 2
# CNN_Kernel_Size = 4
# generator = Generator_CNN(2,hidden_dim,CNN_Out_Channel,CNN_Kernel_Size)
# discriminator_ = Discriminator_LSTM(2,hidden_dim)#,CNN_Out_Channel,CNN_Kernel_Size)
# generator.apply(initialize_weight)
# discriminator_.apply(initialize_weight)
# d_optimizer = torch.optim.Adam(discriminator_.parameters(), lr=0.0001)
# g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
# data_save1 = []
# data_save2 = []
# for epoch in range(EPOCH):
#     evaluate_loss_ = 0
#     evaluate_mode_ = 0
#     for step, (batch_v, batch_x) in enumerate(loader):  # for each training step
#         b_x = Variable(batch_x).to(device)
#         b_v = Variable(batch_v).to(device)
#         b_in = torch.cat((b_v,b_x),1)
#         real = torch.cat((b_v,b_v),1)
#         num_batch = batch_v.size(0)
#         real_label = Variable(torch.ones([num_batch,1]))
#         fake_label = Variable(torch.zeros([num_batch,1]))
#
#         d_optimizer.zero_grad()
#         output = discriminator_(real)
#         errD_real = criterion(output, real_label)
#         errD_real.backward()
#
#         u_out = generator(b_in)
#         errD_fake = adversarial_loss(u_out,b_x,b_v,discriminator_,fake_label)
#         errD_fake.backward()
#         errD = errD_real + errD_fake
#         d_optimizer.step()
#
#         g_optimizer.zero_grad()
#         u_out = generator(b_in)
#         errG = adversarial_loss(u_out, b_x, b_v, discriminator_, real_label)
#         errG.backward()
#         g_optimizer.step()
#
#         evaluate_loss_ += evaluate_loss(generator,b_x,b_v)
#         evaluate_mode_ += evaluate_model(generator, b_x, b_v)
#
#     print(epoch,evaluate_loss_.item()/step,evaluate_mode_.item()/step)
#     data_save1.append(evaluate_loss_.item()/step)
#     data_save2.append(evaluate_mode_.item()/step)
#
# data_save1 = np.asarray(data_save1)
# data_save2 = np.asarray(data_save2)
# savemat('GCDL_loss.mat', {'GCDL_loss':data_save1})
# savemat('GCDL_u.mat', {'GCDL_u':data_save2})
# #
# #
# ##8 GLDL
# print("8 GLDL")
# criterion = torch.nn.BCELoss()
# adversarial_loss = Adversarial_Loss(nominal_controller,plant)
# evaluate_loss = Evaluate_Loss(nominal_controller,plant)
# evaluate_model = Evaluate_Model(nominal_controller,plant)
# hidden_dim = 10
# CNN_Out_Channel = 2
# CNN_Kernel_Size = 4
# generator = Generator_LSTM(2,hidden_dim)#,CNN_Out_Channel,CNN_Kernel_Size)
# discriminator_ = Discriminator_LSTM(2,hidden_dim)#,CNN_Out_Channel,CNN_Kernel_Size)
# generator.apply(initialize_weight)
# discriminator_.apply(initialize_weight)
# d_optimizer = torch.optim.Adam(discriminator_.parameters(), lr=0.0001)
# g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
# data_save1 = []
# data_save2 = []
# for epoch in range(EPOCH):
#     evaluate_loss_ = 0
#     evaluate_mode_ = 0
#     for step, (batch_v, batch_x) in enumerate(loader):  # for each training step
#         b_x = Variable(batch_x).to(device)
#         b_v = Variable(batch_v).to(device)
#         b_in = torch.cat((b_v,b_x),1)
#         real = torch.cat((b_v,b_v),1)
#         num_batch = batch_v.size(0)
#         real_label = Variable(torch.ones([num_batch,1]))
#         fake_label = Variable(torch.zeros([num_batch,1]))
#
#         d_optimizer.zero_grad()
#         output = discriminator_(real)
#         errD_real = criterion(output, real_label)
#         errD_real.backward()
#
#         u_out = generator(b_in)
#         errD_fake = adversarial_loss(u_out,b_x,b_v,discriminator_,fake_label)
#         errD_fake.backward()
#         errD = errD_real + errD_fake
#         d_optimizer.step()
#
#         g_optimizer.zero_grad()
#         u_out = generator(b_in)
#         errG = adversarial_loss(u_out, b_x, b_v, discriminator_, real_label)
#         errG.backward()
#         g_optimizer.step()
#
#         evaluate_loss_ += evaluate_loss(generator,b_x,b_v)
#         evaluate_mode_ += evaluate_model(generator, b_x, b_v)
#
#     print(epoch,evaluate_loss_.item()/step,evaluate_mode_.item()/step)
#     data_save1.append(evaluate_loss_.item()/step)
#     data_save2.append(evaluate_mode_.item()/step)
#
# data_save1 = np.asarray(data_save1)
# data_save2 = np.asarray(data_save2)
# savemat('GLDL_loss.mat', {'GLDL_loss':data_save1})
# savemat('GLDL_u.mat', {'GLDL_u':data_save2})
# #
# #
# #9 GLDC
# print("9 GLDC")
# criterion = torch.nn.BCELoss()
# adversarial_loss = Adversarial_Loss(nominal_controller,plant)
# evaluate_loss = Evaluate_Loss(nominal_controller,plant)
# evaluate_model = Evaluate_Model(nominal_controller,plant)
# hidden_dim = 10
# CNN_Out_Channel = 2
# CNN_Kernel_Size = 4
# generator = Generator_LSTM(2,hidden_dim)#,CNN_Out_Channel,CNN_Kernel_Size)
# discriminator_ = Discriminator_CNN(2,hidden_dim,CNN_Out_Channel,CNN_Kernel_Size)
# generator.apply(initialize_weight)
# discriminator_.apply(initialize_weight)
# d_optimizer = torch.optim.Adam(discriminator_.parameters(), lr=0.0001)
# g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
#
# data_save1 = []
# data_save2 = []
# for epoch in range(EPOCH):
#     evaluate_loss_ = 0
#     evaluate_mode_ = 0
#     for step, (batch_v, batch_x) in enumerate(loader):  # for each training step
#         b_x = Variable(batch_x).to(device)
#         b_v = Variable(batch_v).to(device)
#         b_in = torch.cat((b_v,b_x),1)
#         real = torch.cat((b_v,b_v),1)
#         num_batch = batch_v.size(0)
#         real_label = Variable(torch.ones([num_batch,1]))
#         fake_label = Variable(torch.zeros([num_batch,1]))
#
#         d_optimizer.zero_grad()
#         output = discriminator_(real)
#         errD_real = criterion(output, real_label)
#         errD_real.backward()
#
#         u_out = generator(b_in)
#         errD_fake = adversarial_loss(u_out,b_x,b_v,discriminator_,fake_label)
#         errD_fake.backward()
#         errD = errD_real + errD_fake
#         d_optimizer.step()
#
#         g_optimizer.zero_grad()
#         u_out = generator(b_in)
#         errG = adversarial_loss(u_out, b_x, b_v, discriminator_, real_label)
#         errG.backward()
#         g_optimizer.step()
#
#         evaluate_loss_ += evaluate_loss(generator,b_x,b_v)
#         evaluate_mode_ += evaluate_model(generator, b_x, b_v)
#
#     print(epoch,evaluate_loss_.item()/step,evaluate_mode_.item()/step)
#     data_save1.append(evaluate_loss_.item()/step)
#     data_save2.append(evaluate_mode_.item()/step)
#
# data_save1 = np.asarray(data_save1)
# data_save2 = np.asarray(data_save2)
# savemat('GLDC_loss.mat', {'GLDC_loss':data_save1})
# savemat('GLDC_u.mat', {'GLDC_u':data_save2})