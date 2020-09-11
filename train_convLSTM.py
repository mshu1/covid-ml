import argparse
import os
import shutil
import time
import pdb
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


# import torchvision.models as models
from PIL import Image

print(torch.__version__)
print(torch.cuda.device_count())

import dataloader as dl
import models

args={}
kwargs={}
args['batch_size']=10
args['iter_size']=2000
args['epochs']=3  #The number of Epochs is the number of times you go through the full dataset. 
args['lr']=0.01 #Learning rate is how fast it will decend. 
args['momentum']=0.5 #SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).

args['seed']=1 #random seed
args['log_interval']=10
args['cuda']=True
args['addr']="/big_data2/covid/mshu/covid_1/data_npy/"


normalize = transforms.Normalize(mean=[124/255, 116/255, 104/255],
                                     std=[0.229, 0.224, 0.225])
image_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
eps = np.finfo(np.float32).eps.item()
# torch_eps = torch.finfo(torch.float32).eps

# def train_random(data, model, optimizer):
#     model.train()
#     loss = 0
#     for i in range(args['epochs']):
#         for j in range(args['iter_size']):
#             random_data = random_filter(data)
#             epoch_loss = train_cycle(j, random_data, model, optimizer)
#             loss += epoch_loss
#             if j % 10 == 9:
#                 print('Train Epoch: {}/{} Iter: {}/{} Loss: {:.6f}'.format(
#                 i, args['epochs'], int(j/10), int(args['iter_size'])/10, loss/10))
#                 loss = 0

def train_strat(data, model, optimizer):
    model.train()
    loss = 0
    X, E, T = data['X_ID'], data['E'], data['T']
    E_pos = np.where(E == 1)
    E_neg = np.where(E == 0)
    E_pos_num = int(np.size(E_pos)/np.size(E) * args['batch_size'])
    E_neg_num = args['batch_size'] - E_pos_num
    for i in range(args['epochs']):
        for j in range(args['iter_size']):
            random_data = stratified_filter(data, E_pos, E_neg, E_pos_num, E_neg_num)
            epoch_loss = train_cum_cycle(j, random_data, model, optimizer)
            loss += epoch_loss
            print(epoch_loss)
            if j % 10 == 9:
                print('Train Epoch: {}/{} Iter: {}/{} Loss: {:.6f}'.format(
                i, args['epochs'], int(j/10), int(args['iter_size'])/10, loss/10))
                loss = 0

# def train_cycle(epoch, data, model, optimizer):
#     optimizer.zero_grad()
#     X, E, T, L = data['X'], data['E'], data['T'], data['L']
#     E, L = torch.tensor(E), torch.tensor(L).float().unsqueeze(-1)
#     X_t = read_images_to_batch(args['addr'], X)
#     if args['cuda']:
#         E = E.cuda()
#         X_t = X_t.cuda()
#         L_t = L.cuda()
#     output = model([X_t, L_t])
#     # loss = -(torch.sum(E * (stacked_tensor - torch.log(torch.cumsum(torch.exp(stacked_tensor), dim=0)))))/torch.sum(E, dim=0)[0]
#     neg_loss_per_instance = output - torch.log(torch.cumsum(torch.exp(output), axis=0))
#     loss = -torch.sum(E*neg_loss_per_instance)/(torch.sum(E)+torch.finfo(torch.float32).eps)
#     if (torch.isnan(loss)):
#         print('nan detected')
#         exit()
#     loss.backward()
#     optimizer.step()
#     return loss.detach().cpu().numpy()

def train_cum_cycle(epoch, data, model, optimizer):
    optimizer.zero_grad()
    X, E, T, L = data['X'], data['E'], data['T'], data['L']
    E, L = torch.tensor(E), torch.tensor(L).float().unsqueeze(-1)
    output_list = []
    for i, x_list in enumerate(X):
        X_t = read_npy_to_batch(args['addr'], x_list).unsqueeze(0)
        L_t = L[i,:].unsqueeze(0)
        if args['cuda']:
        # E = E.cuda()
            X_t = X_t.cuda(1)
            L_t = L_t.cuda(1)
        o = model([X_t, L_t])
        output_list.append(o)
        # print(output)
        # exit()
    output = torch.stack(output_list, dim=0)
    if args['cuda']:
        E = E.cuda(0)
    # loss = -(torch.sum(E * (stacked_tensor - torch.log(torch.cumsum(torch.exp(stacked_tensor), dim=0)))))/torch.sum(E, dim=0)[0]
    neg_loss_per_instance = output - torch.log(torch.cumsum(torch.exp(output), axis=0))
    loss = -torch.sum(E*neg_loss_per_instance)/(torch.sum(E)+torch.finfo(torch.float32).eps)
    if (torch.isnan(loss)):
        print('nan detected')
        exit()
    loss.backward()
    optimizer.step()
    del X_t, L_t, E
    torch.cuda.empty_cache()
    return loss.detach().cpu().numpy()


def split_parts(data, n):
  return [data[x:x+n] for x in range(0, len(data), n)]

def read_images_to_batch(path, id_list):
    images_t_list = []
    for id in id_list:
        img_path = os.path.join(path, (id + '.jpg'))
        image = im = Image.open(img_path).convert("RGB")
        # resize to tensors
        images_t_list.append(image_transform(image))
    batch = torch.stack(images_t_list)
    return batch

def read_npy_to_batch(path, id_list):
    images_t_list = []
    for id in id_list:
        img_path = os.path.join(path, (id + '.npy'))
        with open(img_path, 'rb') as f:
            a = np.load(f)
        img_3ch = np.stack([a]*3, -1)
        image = Image.fromarray(img_3ch)
        # resize to tensors
        images_t_list.append(image_transform(image))
    batch = torch.stack(images_t_list)
    return batch

def random_filter(data):
    X, E, T = data['X_ID'], data['E'], data['T']
    batch_indices = np.random.choice(range(len(data['X_ID'])), size=args['batch_size'], replace=False)
    batch_indices.sort()
    X, E, T = X[batch_indices], E[batch_indices], T[batch_indices]
    data_filtered = {'X': X,
                    'E': E,
                    'T': T}
    return data_filtered

def stratified_filter(data, E_pos, E_neg, E_pos_num, E_neg_num):
    X, E, T, L = data['X_ID'], data['E'], data['T'], data['X_LSTM']
    pos_indices = np.random.choice(range(np.size(E_pos)), size=E_pos_num, replace=False)
    neg_indices = np.random.choice(range(np.size(E_neg)), size=E_neg_num, replace=False)
    batch_indices = np.concatenate([E_pos[0][pos_indices], E_neg[0][neg_indices]])
    batch_indices.sort()
    X, E, T, L = X[batch_indices], E[batch_indices], T[batch_indices], L[batch_indices]
    data_filtered = {'X': X,
                    'E': E,
                    'T': T,
                    'L': L}
    return data_filtered

def DDP_Setup():
    world_size = 2
    os.environ['MASTER_ADDR'] = '10.57.23.164'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=3, args=(args,))

def main():
    # model = models.resnet18(pretrained=True)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 1)
    model = models.ConvLSTMNet_multiGPU()
    model = nn.DataParallel(model, device_ids=[0, 1])
    # model = DDP(model)
    model.cuda()
    # optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    data = dl.read_pickle('Admit_Date.pickle')
    train_strat(data, model, optimizer)

def test():
    model = models.ConvLSTMNet_multiGPU()
    model = nn.DataParallel(model)
    # model = DDP(model)
    model.cuda()
    # optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    # data = dl.read_pickle('Admit_Date.pickle')
    # train_strat(data, model, optimizer)
    test = []
    for i in range(1):
        test_t  = torch.ones([60, 2, 3, 224, 224]).cuda()
        ltest_t  = torch.ones([60, 4, 1]).cuda()
        output1 = model([test_t, ltest_t])
        output2 = model([test_t, ltest_t])
        print(output1.shape)
        output = torch.cat([output1, output2], dim=0)
        print(output.shape)
        test.append(output)

test()

