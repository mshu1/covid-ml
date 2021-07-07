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
    # hidden_dim = 15,
    max_days = 10,
    cuda = True,
    baseline_subtraction = True,
    lr = 1e-3,
    input_dim = 58,
    embedding_options = [110, 4, 3, 3],
    embedding_dim = 2
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
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.hidden_dim = 16 * 4
        self.convlstmFc=nn.Linear(64, 64)

        self.lstm = nn.LSTM(net_opt.embedding_size, net_opt.hidden_dim, batch_first = True)

        self.embedding = nn.Linear(net_opt.input_dim, net_opt.embedding_size)
        self.hidden = None

        self.fix_embedding = nn.ModuleList([nn.Embedding(i,2, padding_idx=0) for i in net_opt.embedding_options])

        self.fc1 = nn.Linear(net_opt.hidden_dim+self.hidden_dim+len(net_opt.embedding_options)*net_opt.embedding_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)


    def forward(self, x):
        i_x, t_x, f_x = x[0], x[1], x[2]
        output = self.clstm(i_x)
        x = self.avgpool(output[1][0][0])
        i_x = torch.flatten(x, 1)
        out_x, self.hidden = None, None
        # (236, 50, 30)
        for i in range(net_opt.max_days):
            out_x = self.embedding(t_x[:,i])
            self.lstm.flatten_parameters()
            out_x, self.hidden = self.lstm(out_x.view(i_x.shape[0], 1, -1), self.hidden)
        out_x = torch.squeeze(out_x, dim=1)

        fix_embedding_list = [self.fix_embedding[i](f_x[:,i]) for i in range(len(net_opt.embedding_options))]
        fix_x = torch.cat(fix_embedding_list, dim=1)
        
        i_x = F.relu(self.convlstmFc(i_x))
        i_x = F.dropout(i_x, training=self.training)

        x = torch.cat((i_x, out_x, fix_x), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x

class LSTM_Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self, convLSTM_fig = None):
        super(LSTM_Net, self).__init__()

        self.lstm = nn.LSTM(net_opt.embedding_size, net_opt.hidden_dim, batch_first = True)

        self.embedding = nn.Linear(net_opt.input_dim, net_opt.embedding_size)
        self.hidden = None


        self.fix_embedding = nn.ModuleList([nn.Embedding(i,2, padding_idx=0) for i in net_opt.embedding_options])

        self.fc1 = nn.Linear(net_opt.hidden_dim+len(net_opt.embedding_options)*net_opt.embedding_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)


    def forward(self, x):
        i_x, t_x, f_x = x[0], x[1], x[2]
        out_x, self.hidden = None, None
        # (236, 50, 30)
        for i in range(net_opt.max_days):
            out_x = self.embedding(t_x[:,i])
            self.lstm.flatten_parameters()
            out_x, self.hidden = self.lstm(out_x.view(t_x.shape[0], 1, -1), self.hidden)
        out_x = torch.squeeze(out_x, dim=1)

        fix_embedding_list = [self.fix_embedding[i](f_x[:,i]) for i in range(len(net_opt.embedding_options))]
        fix_x = torch.cat(fix_embedding_list, dim=1)
        x = torch.cat((out_x, fix_x), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x


class DeepConvSurv(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(DeepConvSurv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv2_drop = nn.Dropout2d()  #Dropout
        self.fc1 = nn.Linear(32, 1)

    def forward(self, x):
        #Convolutional Layer/Pooling Layer/Activation
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) 
        #Convolutional Layer/Dropout/Pooling Layer/Activation
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x = x.view(-1, 32)
        x = self.fc1(x)
        return x

class ConvBase(nn.Module):
    #This defines the structure of the NN.
    def __init__(self, convLSTM_fig = None):
        super(ConvBase, self).__init__()
        self.clstm = models.resnet18(pretrained=True)
        self.linear = nn.Linear(1000, 64)

        self.lstm = nn.LSTM(net_opt.embedding_size, net_opt.hidden_dim, batch_first = True)
        self.hidden_dim = 16 * 4
        self.embedding = nn.Linear(net_opt.input_dim, net_opt.embedding_size)
        self.hidden = None

        self.fix_embedding = nn.ModuleList([nn.Embedding(i,2, padding_idx=0) for i in net_opt.embedding_options])

        self.fc1 = nn.Linear(net_opt.hidden_dim+self.hidden_dim+len(net_opt.embedding_options)*net_opt.embedding_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)


    def forward(self, x):
        i_x, t_x, f_x = x[0].squeeze(), x[1], x[2]
        self.clstm.training = self.training
        i_x = self.clstm(i_x)
        i_x = F.relu(self.linear(i_x))
        i_x = F.dropout(i_x, training=self.training)
        out_x, self.hidden = None, None
        # (236, 50, 30)
        for i in range(net_opt.max_days):
            out_x = self.embedding(t_x[:,i])
            self.lstm.flatten_parameters()
            out_x, self.hidden = self.lstm(out_x.view(i_x.shape[0], 1, -1), self.hidden)
        out_x = torch.squeeze(out_x, dim=1)

        fix_embedding_list = [self.fix_embedding[i](f_x[:,i]) for i in range(len(net_opt.embedding_options))]
        fix_x = torch.cat(fix_embedding_list, dim=1)

        x = torch.cat((i_x, out_x, fix_x), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x

class CNN3DModel(nn.Module):
    def __init__(self):
        super(CNN3DModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(1, 32, 13)
        self.conv_layer2 = self._conv_layer_set(32, 32,7)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c, stride):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(2,stride, stride), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((1, 8, 8)),
        )
        return conv_layer

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        
        return out

class Conv3DFull(nn.Module):
    #This defines the structure of the NN.
    def __init__(self, convLSTM_fig = None):
        super(Conv3DFull, self).__init__()
        self.clstm = CNN3DModel()

        self.lstm = nn.LSTM(net_opt.embedding_size, net_opt.hidden_dim, batch_first = True)
        self.hidden_dim = 16 * 4
        self.embedding = nn.Linear(net_opt.input_dim, net_opt.embedding_size)
        self.hidden = None

        self.fix_embedding = nn.ModuleList([nn.Embedding(i,2, padding_idx=0) for i in net_opt.embedding_options])

        self.fc1 = nn.Linear(net_opt.hidden_dim+self.hidden_dim+len(net_opt.embedding_options)*net_opt.embedding_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)


    def forward(self, x):
        i_x, t_x, f_x = x[0].squeeze().unsqueeze(1), x[1], x[2]
        i_x = self.clstm(i_x)
        out_x, self.hidden = None, None
        # (236, 50, 30)
        for i in range(net_opt.max_days):
            out_x = self.embedding(t_x[:,i])
            self.lstm.flatten_parameters()
            out_x, self.hidden = self.lstm(out_x.view(i_x.shape[0], 1, -1), self.hidden)
        out_x = torch.squeeze(out_x, dim=1)

        fix_embedding_list = [self.fix_embedding[i](f_x[:,i]) for i in range(len(net_opt.embedding_options))]
        fix_x = torch.cat(fix_embedding_list, dim=1)

        x = torch.cat((i_x, out_x, fix_x), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x


class ConvCute(nn.Module):
    #This defines the structure of the NN.
    def __init__(self, convLSTM_fig = None):
        super(ConvCute, self).__init__()

        self.lstm = nn.LSTM(net_opt.embedding_size, net_opt.hidden_dim, batch_first = True)
        self.hidden_dim = 2 * 4
        self.embedding = nn.Linear(net_opt.input_dim, net_opt.embedding_size)
        self.hidden = None

        self.fix_embedding = nn.ModuleList([nn.Embedding(i,2, padding_idx=0) for i in net_opt.embedding_options])
        self.fix_cute = nn.Embedding(15,4, padding_idx=0)

        self.fc1 = nn.Linear(net_opt.hidden_dim+self.hidden_dim+len(net_opt.embedding_options)*net_opt.embedding_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)


    def forward(self, x):
        i_x, t_x, f_x = x[0].long(), x[1], x[2]
        out_x, self.hidden = None, None
        # (236, 50, 30)
        for i in range(net_opt.max_days):
            out_x = self.embedding(t_x[:,i])
            self.lstm.flatten_parameters()
            out_x, self.hidden = self.lstm(out_x.view(i_x.shape[0], 1, -1), self.hidden)
        out_x = torch.squeeze(out_x, dim=1)

        fix_embedding_list = [self.fix_embedding[i](f_x[:,i]) for i in range(len(net_opt.embedding_options))]
        fix_x = torch.cat(fix_embedding_list, dim=1)
        cute_embedding_list = [self.fix_cute(i_x[:,i]) for i in range(2)]
        cute_x = torch.cat(cute_embedding_list, dim=1)

        x = torch.cat((cute_x, out_x, fix_x), dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x

class ConvLSTMNet_ImageOnly(nn.Module):
    #This defines the structure of the NN.
    def __init__(self, convLSTM_fig = None):
        super(ConvLSTMNet_ImageOnly, self).__init__()
        self.clstm = convlstm.ConvLSTM(input_dim=1, hidden_dim=[32, 16], kernel_size=(7, 7), num_layers=2, batch_first=True, bias=True,return_all_layers=False)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.hidden_dim = 16 * 4
        self.fc1 = nn.Linear(self.hidden_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)


    def forward(self, x):
        i_x, t_x, f_x = x[0], x[1], x[2]
        output = self.clstm(i_x)
        x = self.avgpool(output[1][0][0])
        i_x = torch.flatten(x, 1)
        x = F.relu(self.fc1(i_x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x

