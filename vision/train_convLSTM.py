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
args['batch_size']=60
args['iter_size']=20 # the number of fix batches/random batches
args['epochs']=2000  #The number of Epochs is the number of times you go through the full dataset. 
args['lr']=0.001 #Learning rate is how fast it will decend. 
args['momentum']=0.5 #SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).

args['seed']=1 #random seed
args['log_interval']=10
args['cuda']=True
args['addr']="/big_data2/covid/mshu/covid_1/data_npy/"


normalize = transforms.Normalize(mean=[124/255],
                                     std=[0.229])
image_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
eps = np.finfo(np.float32).eps.item()

def train_strat(data, model, optimizer):
    model.train()
    loss = 0
    X, E, T = data['X_ID'], data['E'], data['T']
    E_pos = np.where(E == 1)
    E_neg = np.where(E == 0)
    E_pos_num = int(np.size(E_pos)/np.size(E) * args['batch_size'])
    E_neg_num = args['batch_size'] - E_pos_num
    for i in range(args['epochs']):
        djust_learning_rate(optimizer, i)
        for j in range(args['iter_size']):
            random_data = stratified_filter(data, E_pos, E_neg, E_pos_num, E_neg_num)
            # epoch_loss = train_cum_cycle(j, random_data, model, optimizer)
            epoch_loss = train_rand_cycle(j, random_data, model, optimizer, 2)

            loss += epoch_loss
            print(epoch_loss)
            if j % 10 == 9:
                print('Train Epoch: {}/{} Iter: {}/{} Loss: {:.6f}'.format(
                i, args['epochs'], int(j/10), int(args['iter_size'])/10, loss/10))
                loss = 0

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args['lr'] * (0.1 ** (epoch // 2000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_rand_cycle(data, model, optimizer, k, loss_mode = 'Neg', random_images=True):
    model.train()
    optimizer.zero_grad()
    X, E, T, L = data['X'], data['E'], data['T'], data['L']
    E, T, L = torch.tensor(E), torch.tensor(T), torch.tensor(L).float().unsqueeze(-1)
    output_list = []
    batch_list = []
    for i, x_list in enumerate(X):
        if random_images:
            batch_list.append(read_fix_batch(args['addr'], x_list, k, random_images, None))
        else:
            batch_list.append(read_fix_batch(args['addr'], x_list, k, random_images, data['inds'][i]))
    X_t = torch.stack(batch_list)
    if args['cuda']:
        E = E.cuda()
        X_t = X_t.cuda()
        L_t = L.cuda()
        T = T.cuda()
    output = model([X_t, L_t])
    print(output)
    if loss_mode == 'Efron':
        loss = Efron_Loss(T, E, output)
        print(loss)
    else:
        neg_loss_per_instance = output - torch.log(torch.cumsum(torch.exp(output), axis=0))
        loss = -torch.sum(E*neg_loss_per_instance)/(torch.sum(E)+torch.finfo(torch.float32).eps)
    if (torch.isnan(loss)):
        print('nan detected')
        exit()
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().numpy()


def eval_rand_cycle(data, model, k, loss_mode = 'Neg', random_images=True):
    model.eval()
    X, E, T, L = data['X'], data['E'], data['T'], data['L']
    E, T, L = torch.tensor(E), torch.tensor(T), torch.tensor(L).float().unsqueeze(-1)
    output_list = []
    batch_list = []
    for i, x_list in enumerate(X):
        if random_images:
            batch_list.append(read_fix_batch(args['addr'], x_list, k, random_images, None))
        else:
            batch_list.append(read_fix_batch(args['addr'], x_list, k, random_images, data['inds'][i]))
    X_t = torch.stack(batch_list)
    if args['cuda']:
        E = E.cuda()
        X_t = X_t.cuda()
        L_t = L.cuda()
        T = T.cuda()
    output = model([X_t, L_t])
    print(output)
    if loss_mode == 'Efron':
        loss = Efron_Loss(T, E, output)
    else:
        neg_loss_per_instance = output - torch.log(torch.cumsum(torch.exp(output), axis=0))
        loss = -torch.sum(E*neg_loss_per_instance)/(torch.sum(E)+torch.finfo(torch.float32).eps)
    if (torch.isnan(loss)):
        print('nan detected')
        exit()
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


def read_fix_batch(path, id_list, k, random, inds=None):
    images_t_list = []
    if len(id_list) > k:
        if not random:
            if inds is None:
                print("Error: must have proper inds if not randomly generated")
                exit()
        else:
            inds = np.random.choice(range(len(id_list)), size=k, replace=False)
        for ind in inds:
            img_path = os.path.join(path, (id_list[ind] + '.npy'))
            with open(img_path, 'rb') as f:
                a = np.load(f)
            image = Image.fromarray(a)
            # resize to tensors
            images_t_list.append(image_transform(image))   
        return torch.stack(images_t_list)
    else:       
        for id in id_list:
            img_path = os.path.join(path, (id + '.npy'))
            with open(img_path, 'rb') as f:
                a = np.load(f)
            image = Image.fromarray(a)
            # resize to tensors
            images_t_list.append(image_transform(image))
        for i in range(k - len(id_list)):
            images_t_list.append(torch.zeros([1, 224, 224]))
        return torch.stack(images_t_list)


def train_debug(data, model, optimizer):
    model.train()
    loss = 0
    f = 5
    loss_list = []
    for i in range(args['epochs']):
        # adjust_learning_rate(optimizer, i)
        for j in range(args['iter_size']):
            sub_data = data[j]
            if j == 0:
                eval_loss = eval_rand_cycle(sub_data, model, 2, loss_mode='Efron', random_images=False)
                print('Eval Epoch: {}/{} Iter: {}/{} Loss: {:.6f}'.format(
                i, args['epochs'], 0, 0, eval_loss))
                continue
            epoch_loss = train_rand_cycle(sub_data, model, optimizer, 2, loss_mode='Efron', random_images=False)
            loss += epoch_loss
            # print(epoch_loss)
            if j % f == f-1:
                print('Train Epoch: {}/{} Iter: {}/{} Loss: {:.6f}'.format(
                i, args['epochs'], int(j/f), int(args['iter_size']/f), loss/f))
                loss_list.append(loss/f)
                loss = 0
        with open('loss.pickle', 'wb') as handle:
            pickle.dump(loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

def train_debug2(data, model, optimizer, k):
    model.train()
    f = 10
    loss = 0
    batch_list = []
    for i, x_list in enumerate(data['X']):
        batch_list.append(read_fix_batch(args['addr'], x_list, k, True, None))
    X_t = torch.stack(batch_list)
    loss_list = []
    for i in range(args['epochs']):
        # adjust_learning_rate(optimizer, i)
        for j in range(args['iter_size']):
            epoch_loss = train_debug_cycle(data, X_t, model, optimizer)
            loss += epoch_loss
            print(epoch_loss)
            if j % f == f-1:
                print('Train Epoch: {}/{} Iter: {}/{} Loss: {:.6f}'.format(
                i, args['epochs'], int(j/f), int(args['iter_size']/f), loss/f))
                loss_list.append(loss/f)
                loss = 0
    with open('loss.pickle', 'wb') as handle:
        pickle.dump(loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def Efron_Loss(T, E, O):
    Oexp = torch.exp(O)
    Oc = torch.cumsum(Oexp, axis=0)
    loss = 0
    groups = torch.unique(T)
    for num in groups:
      Egroup = E[T==num]
      m = torch.sum(Egroup)
      
      tie_risk = torch.sum(Egroup * O[T==num])

      tie_hazard = torch.sum(Egroup * Oexp[T==num])
      cum_hazard = Oc[T==num][-1] #the very last cum harzard is desired
      # print(Oc[T==num], cum_hazard)
      cum_hazard_array = torch.ones(m).cuda() * cum_hazard if args['cuda'] else torch.ones(m) * cum_hazard
      tie_harzard_array = (torch.arange(0,m).float().cuda()/m) * tie_hazard if args['cuda'] else (torch.arange(0,m).float()/m) * tie_hazard
      tie_diff = torch.log(cum_hazard_array - tie_harzard_array)
      group_neg_likelihood = tie_risk - torch.sum(tie_diff)
      # print(tie_risk, tie_diff, num)
      loss += -group_neg_likelihood
    return loss

def train_debug_cycle(data, X_t, model, optimizer):
    optimizer.zero_grad()
    X, E, T, L = data['X'], data['E'], data['T'], data['L']
    E, L = torch.tensor(E), torch.tensor(L).float().unsqueeze(-1)
    output_list = []
    batch_list = []
    if args['cuda']:
        E = E.cuda()
        X_t = X_t.cuda()
        L_t = L.cuda()
    output = model([X_t, L_t])
    neg_loss_per_instance = output - torch.log(torch.cumsum(torch.exp(output), axis=0))
    loss = -torch.sum(E*neg_loss_per_instance)/(torch.sum(E)+torch.finfo(torch.float32).eps)
    if (torch.isnan(loss)):
        print('nan detected')
        exit()
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().numpy()

def main():
    model = models.ConvLSTMNet_multiGPU()
    model = nn.DataParallel(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    # data = dl.read_pickle('Admit_Date.pickle')
    data = dl.prepare_fix_batch(40, 2, 'Admit_Date.pickle', args['batch_size'])
    # train_debug2(data[0], model, optimizer, 2)
    train_debug(data, model, optimizer)


# def test():
#     model = models.ConvLSTMNet_multiGPU()
#     model = nn.DataParallel(model)
#     # model = DDP(model)
#     model.cuda()
#     # optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
#     optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
#     # data = dl.read_pickle('Admit_Date.pickle')
#     # train_strat(data, model, optimizer)
#     test = []
#     for i in range(1):
#         test_t  = torch.ones([60, 2, 1, 224, 224]).cuda()
#         ltest_t  = torch.ones([60, 88, 1]).cuda()
#         output1 = model([test_t, ltest_t])
#         output2 = model([test_t, ltest_t])
#         print(output1.shape)
#         output = torch.cat([output1, output2], dim=0)
#         print(output.shape)
#         test.append(output)

main()

