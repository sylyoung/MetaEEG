
import numpy as np
import argparse
import time
import os
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
from models.ShallowConvNet import ShallowConvNet, ShallowConvNetFeatures, ShallowConvNetClassifier
from EEG_cross_subject_loader import EEG_loader

def data_load(X, y, args):
    dset_loaders = {}
    train_bs = args.batch_size

    sample_idx = torch.from_numpy(np.arange(len(y))).long()
    data_tar = Data.TensorDataset(X, y, sample_idx)

    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False)
    return dset_loaders


def train_target(args):
    #if args.data in ['SEED', 'SEED4']:
    #    X_tar, y_tar = read_seed_test(args)
    #else:
    #    X_tar, y_tar = read_mi_test(args)
    # dset_loaders = data_load(X_tar, y_tar, args)
    data = EEG_loader(test_subj=args.test_subj, dataset=args.data)

    test_x, test_y = data.test_x, data.test_y
    X_tar, y_tar = torch.from_numpy(test_x).unsqueeze_(3).to(
        torch.float32), torch.squeeze(
        torch.from_numpy(test_y), 1).to(torch.long)
    dset_loaders = data_load(X_tar, y_tar, args)

    #if args.bottleneck == 50:
    #    netF, netC = network.backbone_net(args, 100, return_type='y')
    #if args.bottleneck == 64:
    #    netF, netC = network.backbone_net(args, 128, return_type='y')
    #if args.bottleneck == 'EEGNet':
    #    netF, netC = network.backbone_net(args, 22, 256, 4, return_type='z')

    path = './runs/' + str(args.data) + '/' + str(args.backbone) + str(args.data) + '_seed' + str(
        args.pretrain_seed) + '_pretrain_model_test_subj_' + str(args.test_subj) + '_epoch100.pt'
    #path = './runs/' + str(args.data) + '/' + 'modified_cdmaml_' + str(args.data) + '_test_subj_' + str(
    #                i) + '_shots_' + str(
    #                72) + '_meta_lr_' + str(0.001) + '_fast_lr_' + str(
    #                0.001) + '_meta_batch_size_1_adaptation_steps_' + str(1) + str(
    #                args.backbone) + 'withload_num_iterations_'+ str(10)  + 'seed' + str(s) + '.pth'

    if str(args.backbone) == 'EEGNet':
        if str(args.data) == 'MI2':
            a, b, c = 22, 256, 4
        if str(args.data) == 'ERP2':
            a, b, c = 56, 256, 2
        if str(args.data) == 'MI1':
            a, b, c = 15, 384, 2
        if str(args.data) == 'ERP1':
            a, b, c = 16, 32, 2
        model = EEGNet(a, b, c)
        netF = EEGNet_features(a, b, c).cuda()
        netC = EEGNet_classifier(a, b, c).cuda()
    if str(args.backbone) == 'ShallowConvNet':
        if str(args.data) == 'MI2':
            a, b, c = 4, 22, 4160
        if str(args.data) == 'ERP2':
            a, b, c = 2, 56, 4290
        if str(args.data) == 'MI1':
            a, b, c = 2, 15, 6630
        if str(args.data) == 'ERP1':
            a, b, c = 2, 16, 1690 # reduced
        model = ShallowConvNet(a, b, c)
        netF = ShallowConvNetFeatures(a, b, c).cuda()
        netC = ShallowConvNetClassifier(a, b, c).cuda()
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    # print(type(state_dict))
    netF_state_dict = {}
    netC_state_dict = {}
    for key in state_dict:
        print(key)
        if key.startswith('fc1') or key.startswith('fc'):
            netC_state_dict[key] = state_dict[key]
        else:
            netF_state_dict[key] = state_dict[key]

    netF.load_state_dict(netF_state_dict)
    netC.load_state_dict(netC_state_dict)

    #modelpath = args.output_dir_src + '/source_F.pt'
    #netF.load_state_dict(torch.load(modelpath))
    #modelpath = args.output_dir_src + '/source_C.pt'
    #netC.load_state_dict(torch.load(modelpath))
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

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            mem_label = obtain_label(dset_loaders["Target"], netF, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        features_test = netF(inputs_test)
        outputs_test = netC(features_test)

        # # loss definition
        if args.cls_par > 0:
            pred = mem_label[tar_idx].long()
            #class_weight = torch.tensor([1, 2.42], dtype=torch.float32).cuda()
            #criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            #criterion = torch.nn.CrossEntropyLoss()
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss += gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            acc_t_te = cal_acc(dset_loaders["Target"], netF, netC)
            #acc_t_te = cal_auc_roc(dset_loaders["Target"], netF, netC)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, iter_num, max_iter, acc_t_te)
            print(log_str)
            netF.train()

    if iter_num == max_iter:
        print('{}, TL Acc = {:.2f}%'.format(args.task_str, acc_t_te))
        return acc_t_te


def obtain_label(loader, netF, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netF(inputs)
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
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
    # print(labelset)

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

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    # log_str = 'SSL_Acc = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    # print(log_str)

    return pred_label.astype('int')


if __name__ == '__main__':
    total_acc_arr = []
    total_std_arr = []
    for i in range(1, 10):
        sub_acc_all = []
        for s in range(0, 10):

            test_subj = i
            pretrain_seed = s
            model_name = 'ShallowConvNet'
            data_name_list = ['MI2']

            # data_idx = 2

            for dt in range(1):
                data_idx = dt

                data_name = data_name_list[data_idx]
                if data_name == '001-2014': N, chn, class_num, trial_num = 9, 22, 4, 288  # 001-2014
                if data_name == '001-2014_2': N, chn, class_num, trial_num = 9, 22, 2, 144  # 001-2014_2
                if data_name == 'SEED': N, chn, class_num, trial_num = 15, 62, 3, 3394
                if data_name == 'SEED4': N, chn, class_num, trial_num = 15, 62, 4, 851
                if data_name == 'MI1': N, chn, class_num, trial_num = 1, 15, 2, 100  # added MI2
                if data_name == 'MI2': N, chn, class_num, trial_num = 1, 22, 4, 576  # added MI2
                if data_name == 'ERP1': N, chn, class_num, trial_num = 1, 16, 2, 575  # added MI2
                if data_name == 'ERP2': N, chn, class_num, trial_num = 1, 56, 2, 340  # added MI2

                args = argparse.Namespace(bottleneck=model_name, lr=0.01, lr_decay1=0.1, lr_decay2=1.0, ent=True,
                                          gent=True, cls_par=0.2, ent_par=1.0, epsilon=1e-05, layer='wn', interval=5,
                                          N=N, chn=chn, class_num=class_num, cov_type='oas', trial=trial_num,
                                          threshold=0, distance='cosine')

                args.test_subj = test_subj  # added
                args.pretrain_seed = pretrain_seed  # added
                args.backbone = model_name #args.backbone = 'Net_ln2'

                args.data = data_name
                args.method = 'shot'
                if args.data in ['SEED', 'SEED4']:
                    args.batch_size = 32
                    args.max_epoch = 10
                    args.input_dim = 310
                    args.norm = 'zscore'
                    args.validation = 'random'
                else:
                    args.batch_size = 8 # modified
                    args.max_epoch = 10
                    args.input_dim = int(args.chn * (args.chn + 1) / 2)
                    args.validation = 'last'

                os.environ["CUDA_VISIBLE_DEVICES"] = '0'
                args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
                args.data_env = 'gpu'
                if args.data_env == 'gpu':
                    print('using gpu...')
                else:
                    print('not using gpu...')
                args.SEED = 2020
                #fix_random_seed(args.SEED)
                torch.backends.cudnn.deterministic = True

                #args.data = data_name
                args.output_src = './runs/' + args.data + '/'
                print(args.data)
                print(args)

                #args.local_dir = 'C:/wzw/研一下/0_MSDT/Source_combined/'
                #args.result_dir = 'result/'
                #my_log = LogRecord(args)
                #my_log.log_init()
                #my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

                 #sub_acc_all = np.zeros(N)
                for idt in range(N):
                    args.idt = idt
                    source_str = 'Except_S' + str(test_subj)
                    target_str = 'S' + str(test_subj)
                    info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                    print(info_str)
                    #my_log.record(info_str)
                    #args.log = my_log
                    args.task_str = source_str + '_' + target_str

                    #args.src = ['S' + str(i + 1) for i in range(N)]
                    #args.src.remove(target_str)
                    args.output_dir_src = osp.join(args.output_src, source_str)

                    sub_acc_all.append(train_target(args))#sub_acc_all[idt] = train_target(args)
        print('Sub acc: ', np.round(sub_acc_all, 5))
        print('Avg acc: ', np.round(np.mean(sub_acc_all), 5))
        total_acc_arr.append(np.round(np.mean(sub_acc_all), 5))
        total_std_arr.append(np.round(np.std(sub_acc_all), 5))
    print(str(data_name) + ' SHOT')
    print('avg_arr: ', total_acc_arr)
    print('std_arr: ', total_std_arr)
    print('total_avg: ', np.round(np.mean(total_acc_arr), 5))
    print('total_std: ', np.round(np.std(total_acc_arr), 5))
                #args.log.record("\n==========================================")
                #args.log.record(acc_sub_str)
                #args.log.record(acc_mean_str)

                # record sub acc to csv
                #args.file_str = os.path.basename(__file__).split('.')[0]
                #csv_log = CsvRecord(args)
                #csv_log.init()
                #csv_log.record(sub_acc_all)

                # loss_all = 0.1 * consistency_loss + instance_entropy_loss + batch_entropy_loss + supervised_loss
                # Sub acc:  [84.722 53.472 95.139 76.389 59.028 68.056 72.222 93.75  81.944]
                # Avg acc:  76.08

                # batch size太大了不好，8>4>16
                # 数据对齐操作几乎没有啥效果，可以不用再提了
