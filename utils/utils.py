# -*- coding: utf-8 -*-
# @Time    : 2021/12/20 1:06 下午
# @Author  : wenzhang
# @File    : utils.py
import os.path as osp
import os
import numpy as np
import random
import torch as tr
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def fix_random_seed(SEED):
    tr.manual_seed(SEED)
    tr.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def create_folder(dir_name, data_env, win_root):
    if not osp.exists(dir_name):
        os.system('mkdir -p ' + dir_name)
    if not osp.exists(dir_name):
        if data_env == 'gpu':
            os.mkdir(dir_name)
        elif data_env == 'local':
            os.makedirs(win_root + dir_name)


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def lr_scheduler_full(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cal_auc_roc(loader, netF, netC):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0].cuda()
            labels = data[1].float()
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    #accuracy = tr.sum(tr.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    print('using roc_auc')
    #print(all_label.shape, tr.squeeze(predict).float().shape)
    score = roc_auc_score(all_label, tr.squeeze(predict).float())

    return score * 100

def cal_acc(loader, netF, netC):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0].cuda()
            labels = data[1].float()
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    accuracy = tr.sum(tr.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    return accuracy * 100


def cal_acc_comb(loader, model, flag=True, fc=None):
    print('here1')
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    feas, outputs = model(inputs)
                    outputs = fc(feas)
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    predict = predict.float().numpy()
    all_label = all_label.numpy()
    try:
        auc = roc_auc_score(all_label, predict)
    except ValueError:
        pass
    tn, fp, fn, tp = confusion_matrix(all_label, predict).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)

    return acc * 100, sen * 100, spec * 100, auc * 100


def cal_acc_multi(loader, netF_list, netC_list, args, weight_epoch=None, netG_list=None):
    print('here2')
    num_src = len(netF_list)
    for i in range(len(netF_list)): netF_list[i].eval()

    if args.use_weight:
        if args.method == 'msdt':
            domain_weight = weight_epoch.detach()
            # tmp_weight = np.round(tr.squeeze(domain_weight, 0).t().cpu().detach().numpy().flatten(), 3)
            # print('\ntest domain weight: ', tmp_weight)
    else:
        domain_weight = tr.Tensor([1 / num_src] * num_src).reshape([1, num_src, 1]).cuda()

    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs, labels = data[0].cuda(), data[1]

            if args.use_weight:
                if args.method == 'decision':
                    weights_all = tr.ones(inputs.shape[0], len(args.src))
                    tmp_output = tr.zeros(len(args.src), inputs.shape[0], args.class_num)
                    for i in range(len(args.src)):
                        tmp_output[i] = netC_list[i](netF_list[i](inputs))
                        weights_all[:, i] = netG_list[i](tmp_output[i]).squeeze()
                    z = tr.sum(weights_all, dim=1) + 1e-16
                    weights_all = tr.transpose(tr.transpose(weights_all, 0, 1) / z, 0, 1)
                    weights_domain = tr.sum(weights_all, dim=0) / tr.sum(weights_all)
                    domain_weight = weights_domain.reshape([1, num_src, 1]).cuda()

            outputs_all = tr.cat([netC_list[i](netF_list[i](inputs)).unsqueeze(1) for i in range(num_src)], 1).cuda()
            preds = tr.softmax(outputs_all, dim=2)
            outputs_all_w = (preds * domain_weight).sum(dim=1).cuda()

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    _, predict = tr.max(all_output, 1)
    predict = predict.float().numpy()
    all_label = all_label.numpy()
    try:
        auc = roc_auc_score(all_label, predict)
    except ValueError:
        pass
    tn, fp, fn, tp = confusion_matrix(all_label, predict).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)

    for i in range(len(netF_list)): netF_list[i].train()

    return acc * 100, sen * 100, spec * 100, auc * 100



def data_loader(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    dset_loaders = {}
    train_bs = args.batch_size

    if Xs != None:
        # 随机打乱会导致训练结果偏高，不影响测试
        src_idx = np.arange(len(Ys.numpy()))
        if args.validation == 'random':  # for SEED
            num_train = int(0.9 * len(src_idx))
            tr.manual_seed(args.SEED)
            id_train, id_val = tr.utils.data.random_split(src_idx, [num_train, len(src_idx) - num_train])
        if args.validation == 'last':  # for MI
            num_all = args.trial
            num_train = int(0.9 * num_all)
            id_train = np.array(src_idx).reshape(-1, num_all)[:, :num_train].reshape(1, -1).flatten()
            id_val = np.array(src_idx).reshape(-1, num_all)[:, num_train:].reshape(1, -1).flatten()

        data_src = Data.TensorDataset(Xs, Ys)
        source_tr = Data.TensorDataset(Xs[id_train, :], Ys[id_train])
        source_te = Data.TensorDataset(Xs[id_val, :], Ys[id_val])
    if Xt != None:
        data_tar = Data.TensorDataset(Xt, Yt)

    # for DNN
    if Xs != None:
        dset_loaders["source_tr"] = Data.DataLoader(source_tr, batch_size=train_bs, shuffle=True, drop_last=True)
        dset_loaders["source_te"] = Data.DataLoader(source_te, batch_size=train_bs, shuffle=False, drop_last=False)

    # for DAN/DANN/CDAN/MCC
    if Xs != None:
        dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=True, drop_last=True)
    if Xt != None:
        dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True, drop_last=True)

    # for generating feature
    if Xs != None:
        dset_loaders["Source"] = Data.DataLoader(data_src, batch_size=train_bs * 3, shuffle=False, drop_last=False)
    if Xt != None:
        dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    return dset_loaders
