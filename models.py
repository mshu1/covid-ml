import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import convlstm


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d
        
net_opt = ObjectView(dict(
    embedding_size = 15,
    hidden_dim = 30,
    parameters_num = 9,
    max_days = 4,
    cuda = True,
    batch_size = 20,
    baseline_subtraction = True,
    lr = 1e-3,
    input_dim = 1,
    fe_out_dim = 128
))


class ConvLSTMNet(nn.Module):
    #This defines the structure of the NN.
    def __init__(self, convLSTM_fig = None):
        super(ConvLSTMNet, self).__init__()
        self.clstm = convlstm.ConvLSTM(input_dim=3, hidden_dim=[32, 16], kernel_size=(7, 7), num_layers=2, batch_first=True, bias=True,return_all_layers=False)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.hidden_dim = 16 * 4

        self.lstm = nn.LSTM(net_opt.embedding_size, net_opt.hidden_dim, batch_first = True)
        self.embedding = nn.Linear(net_opt.input_dim, net_opt.embedding_size)
        self.hidden = None

        self.fc1 = nn.Linear(net_opt.hidden_dim+self.hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        i_x, t_x = x[0], x[1]
        output = self.clstm(i_x)
        x = self.avgpool(output[1][0][0])
        i_x = torch.flatten(x, 1)
        out_x, self.hidden = None, None
        for i in range(net_opt.max_days):
            out_x = self.embedding(t_x[:,i])
            out_x, self.hidden = self.lstm(out_x.view(net_opt.batch_size, 1, -1), self.hidden)
        out_x = torch.squeeze(out_x, dim=1)
        x = torch.cat((i_x, out_x), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class ConvLSTMNet_multiGPU(nn.Module):
    #This defines the structure of the NN.
    def __init__(self, convLSTM_fig = None):
        super(ConvLSTMNet_multiGPU, self).__init__()
        self.clstm = convlstm.ConvLSTM(input_dim=1, hidden_dim=[32, 16], kernel_size=(7, 7), num_layers=2, batch_first=True, bias=True,return_all_layers=False)
        # self.dev1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.dev2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # self.clstm = i_x.to(self.dev2)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.hidden_dim = 16 * 4

        self.lstm = nn.LSTM(net_opt.embedding_size, net_opt.hidden_dim, batch_first = True)

        self.embedding = nn.Linear(net_opt.input_dim, net_opt.embedding_size)
        self.hidden = None

        self.fc1 = nn.Linear(net_opt.hidden_dim+self.hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        i_x, t_x = x[0], x[1]
        # self.clstm = i_x.to(self.dev2)
        output = self.clstm(i_x)
        x = self.avgpool(output[1][0][0])
        i_x = torch.flatten(x, 1)
        # i_x = i_x.to(self.dev1)
        out_x, self.hidden = None, None
        for i in range(net_opt.max_days):
            out_x = self.embedding(t_x[:,i])
            self.lstm.flatten_parameters()
            out_x, self.hidden = self.lstm(out_x.view(i_x.shape[0], 1, -1), self.hidden)
        out_x = torch.squeeze(out_x, dim=1)
        x = torch.cat((i_x, out_x), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
