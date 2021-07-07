import argparse
import os
import time
import pickle
import numpy as np
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# import torchvision.models as models
from PIL import Image

print(torch.__version__)
print(torch.cuda.device_count())

import dataloader as dl
import models
import eval_convLSTM as eva

args={}
kwargs={}
args['batch_size']=40
args['k']= 3
args['iter_size']=20 # the number of fix batches/random batches
args['epochs']=30  #The number of Epochs is the number of times you go through the full dataset. 
args['lr']=0.001 #Learning rate is how fast it will decend. 
args['momentum']=0.5 #SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).

args['seed']=1 #random seed
args['log_interval']=10
args['cuda']=True
args['type']='discharge'
args['addr']="/big_data2/covid/mshu/covid_1/data_npy/"
args['arch']="models/0305/3Dconv_{}_b{}_k{}_e{}_i{}/".format(args['type'], args['batch_size'], args['k'], args['epochs'], args['iter_size'])
# args['pref']="data/new-data/icu_discharge/"
args['pref']="data/new-data/{}/".format(args['type'])

args['max_T']=10
args['input_dim']=58

args['mask']= False
args['porp']= 0.95

normalize = transforms.Normalize(mean=[124/255],
                                     std=[0.229])
image_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, (0.8, 1.0)),
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


def train_rand_cycle(data, model, optimizer, k, loss_mode = 'Neg', random_images=True):
    model.train()
    optimizer.zero_grad()
    X, E, T, L, F = data['X'], data['E'], data['T'], data['L'], data['F']
    E, T, L, F = torch.tensor(E), torch.tensor(T), torch.tensor(L).float(), torch.tensor(F).long()
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
        F_t = F.cuda()
    output = model([X_t, L_t, F_t])
    if loss_mode == 'Efron':
        loss = Efron_Loss(T, E, output)
    else:
        neg_loss_per_instance = output - torch.log(torch.cumsum(torch.exp(output), axis=0))
        loss = -torch.sum(E*neg_loss_per_instance)/(torch.sum(E)+torch.finfo(torch.float32).eps)
    if (torch.isnan(loss)):
        print('nan detected')
        exit()
    loss.backward()
    optimizer.step()
    return loss.detach().cpu().numpy(), output.detach().cpu().numpy()


def eval_rand_cycle(data, model, k, baseline_dict=None, loss_mode = 'Neg', random_images=True):
    model.eval()
    X, E, T, L = data['X'], data['E'], data['T'], data['L']
    E_t, T_t, L_t = torch.tensor(E), torch.tensor(T), torch.tensor(L).float()
    batch_list = []
    for i, x_list in enumerate(X):
        if random_images:
            batch_list.append(read_fix_batch(args['addr'], x_list, k, random_images, None))
        else:
            batch_list.append(read_fix_batch(args['addr'], x_list, k, random_images, data['inds'][i]))
    X_t = torch.stack(batch_list)
    if args['cuda']:
        E_t = E_t.cuda()
        X_t = X_t.cuda()
        L_t = L_t.cuda()
        T_t = T_t.cuda()
    output = model([X_t, L_t])
    if loss_mode == 'Efron':
        loss = Efron_Loss(T_t, E_t, output)
    else:
        neg_loss_per_instance = output - torch.log(torch.cumsum(torch.exp(output), axis=0))
        loss = -torch.sum(E*neg_loss_per_instance)/(torch.sum(E)+torch.finfo(torch.float32).eps)
    return loss.detach().cpu().numpy(), output.detach().cpu().numpy()

def eval_rand_split_cycle(data, model, k, baseline_dict=None, loss_mode = 'Neg', random_images=True):
    model.eval()
    X, E, T, L, F = data['X'], data['E'], data['T'], data['L'], data['F']
    E_t, T_t = torch.tensor(E), torch.tensor(T)
    X_list = np.array_split(X, np.ceil(len(data['E'])/args['batch_size']), axis=0)
    L_list = np.array_split(L, np.ceil(len(data['E'])/args['batch_size']), axis=0)
    F_list = np.array_split(F, np.ceil(len(data['E'])/args['batch_size']), axis=0)
    inds = data['inds']
    inds_List = np.array_split(inds, np.ceil(len(data['E'])/args['batch_size']), axis=0)

    output_list = []
    for X, L, F, Inds in zip(X_list, L_list, F_list, inds_List):
        L_t = torch.tensor(L).float()
        F_t = torch.tensor(F).long()
        batch_list = []
        for i, x_list in enumerate(X):
            if random_images:
                batch_list.append(read_fix_batch(args['addr'], x_list, k, random_images, None))
            else:
                batch_list.append(read_fix_batch(args['addr'], x_list, k, random_images, Inds[i]))
        X_t = torch.stack(batch_list)
        if args['cuda']:
            X_t = X_t.cuda()
            L_t = L_t.cuda()
            F_t = F_t.cuda()
        output = model([X_t, L_t, F_t])
        output_list.append(output.detach().cpu())
        del output
    output = torch.cat(output_list)
    if args['cuda']:
        E_t = E_t.cuda()
        T_t = T_t.cuda()
        output = output.cuda()
    if loss_mode == 'Efron':
        loss = Efron_Loss(T_t, E_t, output)
    else:
        neg_loss_per_instance = output - torch.log(torch.cumsum(torch.exp(output), axis=0))
        loss = -torch.sum(E*neg_loss_per_instance)/(torch.sum(E)+torch.finfo(torch.float32).eps)
    return loss.detach().cpu().numpy(), output.detach().cpu().numpy()

def train_debug(data, eval_data, test_data, model, optimizer, k=2):
    model.train()
    loss, avg_ci, eval_max_ci, best_epoch = 0,0,0,0
    f = 5
    test_batch = dl.prepare_eval_batch(k, test_data)
    # eval_batch = dl.prepare_eval_batch(k, eval_data)

    loss_list, eval_loss_list, test_loss_list, test_ci_list, test_ev = [], [], [], [], []

    for epoch in range(args['epochs']):
        # Training
        output_total = np.zeros(args['iter_size']*args['batch_size'])
        iter_data, E_total, T_total = dl.prepare_fix_batch(args['iter_size'], k, data, args['batch_size'])
        ci_list = np.zeros(args['iter_size'])
        for j in range(args['iter_size']):
            sub_data = iter_data[j]
            epoch_loss, output = train_rand_cycle(sub_data, model, optimizer, k, loss_mode='Efron', random_images=False)
            loss += epoch_loss
            output_total[j*args['batch_size']:(j+1)*args['batch_size']] = np.squeeze(output)
            baseline_cum_hazard = eva.compute_cum_baseline_hazard(sub_data['T'], sub_data['E'], output)
            ev = eva.predict_results(baseline_cum_hazard, sub_data['T'], sub_data['E'], output)
            ci = ev.concordance_td()
            avg_ci += ci
            ci_list[j] = ci
            if j % f == f-1:
                print('Train Epoch: {}/{} Iter: {}/{} Loss: {:.6f}, CI: {:.6f}'.format(
                    epoch, args['epochs'], int(j/f), int(args['iter_size']/f), loss/f, avg_ci/f))
                loss_list.append(loss/f)
                loss = 0
                avg_ci = 0
        #Eval
        eval_batch = dl.prepare_eval_batch(k, eval_data)
        eval_loss, eval_output = eval_rand_split_cycle(eval_batch, model, k, loss_mode='Efron', random_images=False)
        eval_loss_list.append(eval_loss)        
        max_ci_ind = np.argmax(ci_list)
        baseline_cum_hazard = eva.compute_cum_baseline_hazard(iter_data[max_ci_ind]['T'], 
                            iter_data[max_ci_ind]['E'], 
                            output_total[max_ci_ind*args['batch_size']:(max_ci_ind+1)*args['batch_size']])
        ev = eva.predict_results(baseline_cum_hazard, eval_batch['T'], eval_batch['E'], eval_output)
        eval_ci = ev.concordance_td()
        print('Eval Epoch: {}/{} Iter: {}/{} Loss: {:.6f}, CI: {:.6f}'.format(
            epoch, args['epochs'], 0, 0, eval_loss, eval_ci))

        is_best = eval_ci > eval_max_ci
        eval_max_ci = max(eval_ci, eval_max_ci)
        if is_best: best_epoch = epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args['arch'],
            'state_dict': model.state_dict(),
            'best_ci': eval_max_ci,
            'optimizer' : optimizer.state_dict(),
        }, is_best, prefix=args['arch']+'/')

        # Test
        # test_batch = dl.prepare_eval_batch(k, test_data)
        eval_loss, eval_output = eval_rand_split_cycle(test_batch, model, k, loss_mode='Efron', random_images=False)
        test_loss_list.append(eval_loss)
        max_ci_ind = np.argmax(ci_list)
        baseline_cum_hazard = eva.compute_cum_baseline_hazard(iter_data[max_ci_ind]['T'], 
                            iter_data[max_ci_ind]['E'], 
                            output_total[max_ci_ind*args['batch_size']:(max_ci_ind+1)*args['batch_size']])
        ev = eva.predict_results(baseline_cum_hazard, test_batch['T'], test_batch['E'], eval_output)
        test_ci = ev.concordance_td()
        test_ci_list.append(test_ci)
        test_ev.append(ev)
        print('Test Epoch: {}/{} Iter: {}/{} Loss: {:.6f}, CI: {:.6f}'.format(
            epoch, args['epochs'], 0, 0, eval_loss, test_ci))

        with open(args['arch']+'/loss.pickle', 'wb') as handle:
            pickle.dump(loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


        with open(args['arch']+'/eval_loss.pickle', 'wb') as handle:
            pickle.dump(eval_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args['arch']+'/test_loss.pickle', 'wb') as handle:
            pickle.dump(test_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args['arch']+'/ev_list.pickle', 'wb') as handle:
            pickle.dump(test_ev, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    val_loss_min = np.argmin(np.array(eval_loss_list))
    print(args['arch'], args['mask'], args['porp'])
    print("Training Completed: best performance @ {} with test ci {}".format(val_loss_min, test_ci_list[val_loss_min]))
    print("(CI)Training Completed: best performance @ {} with test ci {}".format(best_epoch, test_ci_list[best_epoch]))



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
    if not os.path.exists(args['arch']):
        os.makedirs(args['arch'])
    print(args['arch'])
    models.net_opt.max_days = args['max_T']
    models.net_opt.input_dim = args['input_dim']
    model = models.Conv3DFull()
    model = nn.DataParallel(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    # data = dl.read_pickle('Admit_Date.pickle')
    # data = dl.prepare_fix_batch(args['iter_size'], 2, 'Admit_Date.pickle', args['batch_size'])
    # train_debug2(data[0], model, optimizer, 2)
    # data = dl.read_pickle('data_process/edit-0108/Intubation_Date_Predict_NEW.pickle', max_T=args['max_T'], admit=False)
    # train_split, eval_split, test_split = dl.make_train_test_split(data)
    # pref = 'data/Discharge_Patched10/'
    print('Mask: ', args['mask'], 'Shadow: ', args['porp'])
    pref = args['pref']
    train_split, eval_split, test_split = dl.load_train_test_split(pref+'train_split.pickle', pref+'eval_split.pickle',pref+'test_split.pickle')
    print(len(train_split['E']),len(test_split['E']), len(eval_split['E']))
    train_debug(train_split, eval_split, test_split, model, optimizer, k=args['k'])


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

# def train_debug2(data, model, optimizer, k):
#     model.train()
#     f = 10
#     loss = 0
#     batch_list = []
#     for i, x_list in enumerate(data['X']):
#         batch_list.append(read_fix_batch(args['addr'], x_list, k, True, None))
#     X_t = torch.stack(batch_list)
#     loss_list = []
#     for i in range(args['epochs']):
#         # adjust_learning_rate(optimizer, i)
#         for j in range(args['iter_size']):
#             epoch_loss = train_debug_cycle(data, X_t, model, optimizer)
#             loss += epoch_loss
#             print(epoch_loss)
#             if j % f == f-1:
#                 print('Train Epoch: {}/{} Iter: {}/{} Loss: {:.6f}'.format(
#                 i, args['epochs'], int(j/f), int(args['iter_size']/f), loss/f))
#                 loss_list.append(loss/f)
#                 loss = 0
#     with open('loss.pickle', 'wb') as handle:
#         pickle.dump(loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

##############################################

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

def image_mask(im):
    h,w = im.shape
    ah, aw = np.floor(args['porp']*h/2), np.floor(args['porp']*w/2)
    ch, cw = np.floor(h/2), np.floor(w/2)
    # im[int(ch-ah):int(ch+ah), int(cw-aw):int(cw+aw)] = 0
    # im[:, int(cw-aw):int(cw+aw)] = 0
    im[:,:] = 0
    return im

def read_fix_batch(path, id_list, k, random, inds=None):
    images_t_list = []
    if len(id_list) > k:
        if not random:
            if inds is None:
                print("Error: must have proper inds if not randomly generated")
                exit()
        else:
            inds = np.random.choice(range(len(id_list)), size=k, replace=False)
            inds.sort()
        for ind in inds:
            img_path = os.path.join(path, (id_list[ind] + '.npy'))
            with open(img_path, 'rb') as f:
                a = np.load(f)
            if args['mask']:
                a = image_mask(a)
            image = Image.fromarray(a)
            image.save('check.jpg')
            # resize to tensors
            images_t_list.append(image_transform(image))   
        return torch.stack(images_t_list)
    else:       
        for id in id_list:
            img_path = os.path.join(path, (id + '.npy'))
            with open(img_path, 'rb') as f:
                a = np.load(f)
            if args['mask']:
                a = image_mask(a)
            image = Image.fromarray(a)
            # resize to tensors
            images_t_list.append(image_transform(image))
        for i in range(k - len(id_list)):
            images_t_list.append(torch.zeros([1, 224, 224]))
        return torch.stack(images_t_list)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args['lr'] * (0.1 ** (epoch // 2000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def Efron_Loss(T, E, O):
    Oexp = torch.exp(O)
    Oc = torch.cumsum(Oexp, axis=0)
    loss = 0
    groups = torch.unique(T)
    for num in groups:
      Egroup = E[T==num]
      m = torch.sum(Egroup)
      
      tie_risk = torch.sum(Egroup * O[T==num].squeeze())

      tie_hazard = torch.sum(Egroup * Oexp[T==num].squeeze())
      cum_hazard = Oc[T==num][-1] #the very last harzard is desired due to ordering
      cum_hazard_array = torch.ones(m).cuda() * cum_hazard if args['cuda'] else torch.ones(m) * cum_hazard
      tie_harzard_array = (torch.arange(0,m).float().cuda()/m) * tie_hazard if args['cuda'] else (torch.arange(0,m).float()/m) * tie_hazard
      tie_diff = torch.log(cum_hazard_array - tie_harzard_array)
      group_neg_likelihood = tie_risk - torch.sum(tie_diff)
      loss += -group_neg_likelihood
    return loss

def save_checkpoint(state, is_best, prefix='', filename='checkpoint.pth.tar'):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')

############################################################

main()

