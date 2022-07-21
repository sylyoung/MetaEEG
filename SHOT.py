import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import os.path as osp

from scipy.spatial.distance import cdist
from torch.utils.data import TensorDataset, DataLoader

from utils import network, loss
from utils.CsvRecord import CsvRecord
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_test, read_seed_test
from utils.utils import lr_scheduler, fix_random_seed, op_copy, cal_acc, cal_auc_roc
from models.EEGNet import EEGNet, EEGNet_features, EEGNet_classifier
from models.ShallowConvNet import ShallowConvNet, ShallowConvNetFeatures, ShallowConvNetClassifier, ShallowConvNetFeaturesReduced, ShallowConvNetReduced
from EEG_cross_subject_loader import EEG_loader

import argparse
import time
import os
import random


def data_load(X, y, args):
    dset_loaders = {}
    train_bs = args.batch_size

    sample_idx = torch.from_numpy(np.arange(len(y))).long()
    data_tar = Data.TensorDataset(X, y, sample_idx)

    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False)
    return dset_loaders


def train_target(args):
    data = EEG_loader(test_subj=args.test_subj, dataset=args.dataset)

    test_x, test_y = data.test_x, data.test_y
    X_tar, y_tar = torch.from_numpy(test_x).unsqueeze_(3).to(
        torch.float32), torch.squeeze(
        torch.from_numpy(test_y), 1).to(torch.long)
    dset_loaders = data_load(X_tar, y_tar, args)

    EEGNet_fc_num = {'MI1': 256, 'MI2': 384, 'ERP1': 32, 'ERP2': 256}
    ShallowConvNet_fc_num = {'MI1': 16640, 'MI2': 26520, 'ERP1': 6760, 'ERP2': 17160}

    if args.mode == 'baseline':
        if args.backbone == 'EEGNet':
            #model = EEGNet(chn, EEGNet_fc_num[args.dataset], class_num).to(device)
            netF = EEGNet_features(chn, EEGNet_fc_num[args.dataset], class_num).to(device)
            netC = EEGNet_classifier(chn, EEGNet_fc_num[args.dataset], class_num).to(device)
        elif args.backbone == 'ShallowConvNet':
            if args.dataset == 'ERP1':
                #model = ShallowConvNetReduced(class_num, chn, ShallowConvNet_fc_num[args.dataset]).to(device)
                netF = ShallowConvNetFeaturesReduced(class_num, chn, ShallowConvNet_fc_num[args.dataset]).to(device)
            else:
                #model = ShallowConvNet(class_num, chn, ShallowConvNet_fc_num[args.dataset]).to(device)
                netF = ShallowConvNetFeatures(class_num, chn, ShallowConvNet_fc_num[args.dataset]).to(device)
            netC = ShallowConvNetClassifier(class_num, chn, ShallowConvNet_fc_num[args.dataset]).to(device)
    model = torch.load(args.path).to(device)
    state_dict = model.state_dict()
    netF_state_dict = {}
    netC_state_dict = {}
    for key in state_dict:
        if key.startswith('classifier_block') or key.startswith('fc'):
            netC_state_dict[key] = state_dict[key]
        else:
            netF_state_dict[key] = state_dict[key]

    netF.load_state_dict(netF_state_dict)
    netC.load_state_dict(netC_state_dict)

    netC.eval()

    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"]) # epoch * batch_number
    interval_iter = max_iter // args.interval
    iter_num = 0

    iter_test = None
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.to(args.device)
        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            mem_label = obtain_label(dset_loaders["Target"], netF, netC, args)
            mem_label = torch.from_numpy(mem_label).to(args.device)
            netF.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        features_test = netF(inputs_test)
        outputs_test = netC(features_test)

        # # loss definition
        if args.cls_par > 0:
            pred = mem_label[tar_idx].long()
            # class_weight = torch.tensor([1, 2.42], dtype=torch.float32).cuda()
            # criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
        else:
            classifier_loss = torch.tensor(0.0).to(args.device)

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            # criterion = torch.nn.CrossEntropyLoss()
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + args.epsilon))
                print('entropy_loss:', round(entropy_loss.item(), 3), 'gentropy_loss:', round(gentropy_loss.item(), 3))
                entropy_loss += gentropy_loss
            im_loss = entropy_loss * args.ent_par
            print('classifier_loss:', round(classifier_loss.item(), 3), 'im_loss', round(im_loss.item(), 3))
            classifier_loss += im_loss
        print('loss', round(classifier_loss.item(), 3))
        print('#' * 20)
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            if args.dataset == 'MI1' or args.dataset == 'MI2':
                acc_t_te = cal_acc(dset_loaders["Target"], netF, netC, args.device)
            else:
                acc_t_te = cal_auc_roc(dset_loaders["Target"], netF, netC, args.device)
            #log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, iter_num, max_iter, acc_t_te)
            #print(log_str)
            netF.train()

    if iter_num == max_iter:
        print('TL Score = {:.2f}%'.format(acc_t_te))
        return acc_t_te


def obtain_label(loader, netF, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            #labels = data[1]
            inputs = inputs.to(args.device)
            feas = netF(inputs)
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                #all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                #all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    #accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):  # SSL
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    #acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    return pred_label.astype('int')


if __name__ == '__main__':

    mode = 'baseline'
    # mode = 'MDMAML'
    #for model_name in ['EEGNet', 'ShallowConvNet']:
    for model_name in ['EEGNet']:
        #for dataset in ['MI1', 'MI2', 'ERP1', 'ERP2']:
        for dataset in ['MI1', 'MI2']:
            print('SHOT', model_name, dataset)

            if dataset == 'MI1':
                subj_num = 9
            elif dataset == 'MI2':
                subj_num = 14
            elif dataset == 'ERP1':
                subj_num = 10
            elif dataset == 'ERP2':
                subj_num = 16

            if dataset == 'MI1':
                ways = 4
            else:
                ways = 2

            total_acc_arr = []
            total_std_arr = []

            for test_subj in range(0, subj_num):
                sub_acc_all = []
                for se in range(0, 10):
                    print('Test Subject', test_subj, 'Seed', se)

                    if dataset == 'MI1':
                        chn, class_num, trial_num = 22, 4, 576
                    if dataset == 'MI2':
                        chn, class_num, trial_num = 15, 2, 100
                    if dataset == 'ERP1':
                        chn, class_num, trial_num = 16, 2, 575
                    if dataset == 'ERP2':
                        chn, class_num, trial_num = 56, 2, 340

                    args = argparse.Namespace(lr=0.01, lr_decay1=0.1, lr_decay2=1.0, ent=True,
                                              gent=True, cls_par=0.3, ent_par=1.0, epsilon=1e-05, layer='wn',
                                              interval=15,
                                              chn=chn, class_num=class_num, cov_type='oas', trial=trial_num,
                                              threshold=0, distance='cosine')

                    args.seed = 42
                    random.seed(args.seed)
                    np.random.seed(args.seed)
                    torch.manual_seed(args.seed)

                    device = torch.device('cpu')
                    args.cuda = True
                    if args.cuda:
                        torch.cuda.manual_seed(args.seed)
                        torch.backends.cudnn.deterministic = True
                        device = torch.device('cuda:0')

                    args.test_subj = test_subj
                    args.pretrain_seed = se
                    args.backbone = model_name
                    args.dataset = dataset
                    args.method = 'shot'
                    args.device = device
                    args.batch_size = 8
                    args.max_epoch = 10
                    args.output_src = './runs/' + args.dataset + '/'
                    args.mode = mode

                    if args.mode == 'baseline':
                        path = './runs/' + str(args.dataset) + '/baseline_' + str(args.backbone) + str(
                            args.dataset) + '_seed' + str(
                            args.pretrain_seed) + '_test_subj_' + str(args.test_subj) + '_epoch100.pt'
                        args.path = path
                    elif args.mode == 'MDMAML':
                        path1 = './runs/' + str(args.dataset) + '/mdmaml_model1_' + str(
                            args.dataset) + '_test_subj_' + \
                                str(args.test_subj) + dataset + '_test_subj_' + str(test_subj) + \
                                '_shots_1_meta_lr_0.001_fast_lr_0.001_meta_batch_size_1_adaptation_steps_1' \
                                + str(model_name) + '_num_iterations_500seed' + str(se) + '.pt'
                        path2 = './runs/' + str(args.dataset) + '/mdmaml_model2_' + str(
                            args.dataset) + '_test_subj_' + \
                                str(args.test_subj) + dataset + '_test_subj_' + str(test_subj) + \
                                '_shots_1_meta_lr_0.001_fast_lr_0.001_meta_batch_size_1_adaptation_steps_1' \
                                + str(model_name) + '_num_iterations_500seed' + str(se) + '.pt'
                        args.paths = [path1, path2]

                    #source_str = 'Except_S' + str(test_subj)
                    #target_str = 'S' + str(test_subj)
                    #info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                    #print(info_str)

                    #args.task_str = source_str + '_' + target_str
                    #args.output_dir_src = osp.join(args.output_src, source_str)

                    sub_acc_all.append(train_target(args))

                print('Sub acc: ', np.round(sub_acc_all, 5))
                print('Avg acc: ', np.round(np.mean(sub_acc_all), 5))
                total_acc_arr.append(np.round(np.mean(sub_acc_all), 5))
                total_std_arr.append(np.round(np.std(sub_acc_all), 5))

            print(str(dataset) + ' SHOT')
            print('avg_arr: ', total_acc_arr)
            print('std_arr: ', total_std_arr)
            print('total_avg: ', np.round(np.mean(total_acc_arr), 5))
            print('total_std: ', np.round(np.std(total_acc_arr), 5))
