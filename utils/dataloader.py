# -*- coding: utf-8 -*-
# @Time    : 2021/12/18 11:04
# @Author  : wenzhang
# @File    : dataloader.py
import torch as tr
import numpy as np
from sklearn import preprocessing
from torch.autograd import Variable
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from imblearn.over_sampling import SMOTE
from utils.data_augment import data_aug
from scipy.io import loadmat, savemat
from os import walk


def read_mi_all(args):
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    data, label = [], []
    for s in range(args.N):
        # each source sub
        src_data = np.squeeze(Data_raw[s, :, :, :])
        src_label = Label[s, :].reshape(-1, 1)

        if args.aug:
            sample_size = src_data.shape[2]
            # mult_flag, noise_flag, neg_flag, freq_mod_flag
            flag_aug = [True, True, True, True]
            src_data = np.transpose(src_data, (0, 2, 1))
            src_data, src_label = data_aug(src_data, src_label, sample_size, flag_aug)
            src_data = np.transpose(src_data, (0, 2, 1))

        covar = Covariances(estimator=args.cov_type).transform(src_data)
        fea_tsm = TangentSpace().fit_transform(covar)
        src_label = src_label.reshape(-1, 1)

        data.append(fea_tsm)
        label.append(src_label)

    return data, label


def read_mi_train(args):
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # source sub
    src_data = np.squeeze(Data_raw[args.ids, :, :, :])
    src_label = np.squeeze(Label[args.ids, :])
    src_label = tr.from_numpy(src_label).long()
    print(src_data.shape, src_label.shape)  # (288, 22, 750)

    if args.aug:
        sample_size = src_data.shape[2]
        # mult_flag, noise_flag, neg_flag, freq_mod_flag
        flag_aug = [True, True, True, True]
        # flag_aug = [True, False, False, False]
        src_data = np.transpose(src_data, (0, 2, 1))
        src_data, src_label = data_aug(src_data, src_label, sample_size, flag_aug)
        src_data = np.transpose(src_data, (0, 2, 1))
        src_label = tr.from_numpy(src_label).long()
    # print(src_data.shape, src_label.shape)  # (288*7, 22, 750)

    covar = Covariances(estimator=args.cov_type).transform(src_data)
    fea_tsm = TangentSpace().fit_transform(covar)
    fea_tsm = Variable(tr.from_numpy(fea_tsm).float())

    # X.shape - (#samples, # feas)
    print(fea_tsm.shape, src_label.shape)

    return fea_tsm, src_label


def read_mi_test(args):
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # target sub
    tar_data = np.squeeze(Data_raw[args.idt, :, :, :])
    tar_label = np.squeeze(Label[args.idt, :])
    tar_label = tr.from_numpy(tar_label).long()

    # 288 * 22 * 750
    covar_src = Covariances(estimator=args.cov_type).transform(tar_data)
    fea_tsm = TangentSpace().fit_transform(covar_src)

    # covar = Covariances(estimator=cov_type).transform(tar_data)
    # tmp_ref = TangentSpace().fit(covar[:ntu, :, :])
    # fea_tsm = tmp_ref.transform(covar)

    fea_tsm = Variable(tr.from_numpy(fea_tsm).float())

    # X.shape - (#samples, # feas)
    print(fea_tsm.shape, tar_label.shape)
    return fea_tsm, tar_label


def read_mi_test_aug(args):
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # target sub
    tar_data = np.squeeze(Data_raw[args.idt, :, :, :])
    tar_label = np.squeeze(Label[args.idt, :])

    # 288 * 22 * 750
    covar_tar = Covariances(estimator=args.cov_type).transform(tar_data)
    X_tar = TangentSpace().fit_transform(covar_tar)
    X_tar = Variable(tr.from_numpy(X_tar).float())
    y_tar = tr.from_numpy(tar_label).long()

    sample_size = tar_data.shape[2]
    flag_aug = [True, True, True, True]
    tar_data_tmp = np.transpose(tar_data, (0, 2, 1))
    tar_data_tmp, tar_label_aug = data_aug(tar_data_tmp, tar_label, sample_size, flag_aug)
    tar_data_aug = np.transpose(tar_data_tmp, (0, 2, 1))

    # 288 * 22 * 750
    covar_tar = Covariances(estimator=args.cov_type).transform(tar_data_aug)
    X_tar_aug = TangentSpace().fit_transform(covar_tar)
    X_tar_aug = Variable(tr.from_numpy(X_tar_aug).float())
    y_tar_aug = tr.from_numpy(tar_label_aug).long()

    # X.shape - (#samples, # feas)
    print(y_tar.shape, y_tar.shape)
    print(X_tar_aug.shape, y_tar_aug.shape)
    return X_tar, y_tar, X_tar_aug, y_tar_aug


def read_mi_combine(args):  # no data augment
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']
    # print('raw data shape', Data_raw.shape, Label.shape)

    Data_new = Data_raw.copy()
    n_sub = len(Data_raw)

    # MTS transfer
    ids = np.delete(np.arange(0, n_sub), args.idt)
    src_data, src_label = [], []
    for i in range(n_sub - 1):
        src_data.append(np.squeeze(Data_new[ids[i]]))
        src_label.append(np.squeeze(Label[ids[i]]))
    src_data = np.concatenate(src_data, axis=0)
    src_label = np.concatenate(src_label, axis=0)

    # final label
    src_label = np.squeeze(src_label)
    src_label = tr.from_numpy(src_label).long()
    print(src_data.shape, src_label.shape)

    # final features
    covar = Covariances(estimator=args.cov_type).transform(src_data)
    fea_tsm = TangentSpace().fit_transform(covar)
    src_data = Variable(tr.from_numpy(fea_tsm).float())

    return src_data, src_label


def read_mi_combine_tar(args):  # no data augment
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']
    print('raw data shape', Data_raw.shape, Label.shape)

    Data_new = Data_raw.copy()
    n_sub = len(Data_raw)

    # combine multiple source data
    ids = np.delete(np.arange(0, n_sub), args.idt)
    src_data, src_label = [], []
    for i in range(n_sub - 1):
        src_data.append(np.squeeze(Data_new[ids[i]]))
        src_label.append(np.squeeze(Label[ids[i]]))
    src_data = np.concatenate(src_data, axis=0)
    src_label = np.concatenate(src_label, axis=0)

    # final label
    src_label = np.squeeze(src_label)
    src_label = tr.from_numpy(src_label).long()
    print(src_data.shape, src_label.shape)  # (n_src, chns, fts)
    covar = Covariances(estimator=args.cov_type).transform(src_data)
    fea_tsm = TangentSpace().fit_transform(covar)  # tangent space transform
    src_data = Variable(tr.from_numpy(fea_tsm).float())  # (n_src, low_dim_fts)

    # single target domain data
    tar_data = np.squeeze(Data_new[args.idt, :, :, :])
    tar_label = np.squeeze(Label[args.idt, :])
    print(tar_data.shape, tar_label.shape)  # (n_tar, chns, fts)
    covar = Covariances(estimator=args.cov_type).transform(tar_data)
    tmp_ref = TangentSpace().fit(covar)  # tangent space transform
    fea_tsm = tmp_ref.transform(covar)
    tar_data = Variable(tr.from_numpy(fea_tsm).float())
    tar_label = tr.from_numpy(tar_label).long()
    print(src_data.shape, src_label.shape)  # (n_src, low_dim_fts)
    print(tar_data.shape, tar_label.shape)  # (n_tar, low_dim_fts)

    return src_data, src_label, tar_data, tar_label


def read_seizure_combine_tar(args):  # no data augment
    # (9, 288, 22, 750) (9, 288)
    if args.data_env == 'local':
        file = '../data/fts_labels/' + args.data + '.npz'
    if args.data_env == 'gpu':
        domains = next(walk('/home/zwwang/code/Source_combined/data/fts_labels/'), (None, None, []))[2]
        i = args.idt
        src_x = []
        src_y = []
        for j in range(len(domains)):
            if i != j:
                src = loadmat('/home/zwwang/code/Source_combined/data/fts_labels/' + domains[j])
                src0, src1 = src['data'], src['label']
                src_x.append(src0)
                src_y.append(src1)
        src_data = np.concatenate(src_x, axis=0)
        src_label = np.concatenate(src_y, axis=1).squeeze()
        # print(src_data.shape, src_label.shape)  # (n_src, chns, fts)
        # load target domain
        tar = loadmat('/home/zwwang/code/Source_combined/data/fts_labels/' + domains[i])
        tar_data, tar_label = tar['data'], tar['label']
        tar_label = tar_label.squeeze()
        # print(tar_data.shape, tar_label.shape)  # (n_tar, chns, fts)

    # smooth class imbalance
    oversample = SMOTE(random_state=42)
    src_data, src_label = oversample.fit_resample(src_data, src_label)
    src_data = data_normalize(src_data, args.norm)
    src_data = Variable(tr.from_numpy(src_data).float())
    src_label = tr.from_numpy(src_label).long()
    print(src_data.shape, src_label.shape)  # (n_src, chns, fts)

    # single target domain data
    tar_data = data_normalize(tar_data, args.norm)
    tar_data = Variable(tr.from_numpy(tar_data).float())
    tar_label = tr.from_numpy(tar_label).long()
    print(tar_data.shape, tar_label.shape)  # (n_tar, chns, fts)

    return src_data, src_label, tar_data, tar_label


def data_normalize(fea_de, norm_type):
    if norm_type == 'zscore':
        zscore = preprocessing.StandardScaler()
        fea_de = zscore.fit_transform(fea_de)

    return fea_de


def read_seed_all(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    data, label = [], []
    for s in range(args.N):
        # each source sub
        fea_de = np.squeeze(Data_raw[s, :, :])
        src_label = Label[s, :].reshape(-1, 1)
        data.append(fea_de)
        label.append(src_label)

    return data, label


def read_seed_train(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # source sub
    fea_de = np.squeeze(Data_raw[args.ids, :, :])
    fea_de = data_normalize(fea_de, args.norm)
    fea_de = Variable(tr.from_numpy(fea_de).float())

    src_label = np.squeeze(Label[args.ids, :])
    src_label = tr.from_numpy(src_label).long()
    print(fea_de.shape, src_label.shape)

    return fea_de, src_label


def read_seed_test(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # target sub
    fea_de = np.squeeze(Data_raw[args.idt, :, :])
    fea_de = data_normalize(fea_de, args.norm)
    fea_de = Variable(tr.from_numpy(fea_de).float())

    tar_label = np.squeeze(Label[args.idt, :])
    tar_label = tr.from_numpy(tar_label).long()
    print(fea_de.shape, tar_label.shape)

    return fea_de, tar_label


def read_seed_combine(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']
    print(Data_raw.shape, Label.shape)

    n_sub = len(Data_raw)
    ids = np.delete(np.arange(0, n_sub), args.idt)
    src_data, src_label = [], []
    for i in range(n_sub - 1):
        src_data.append(np.squeeze(Data_raw[ids[i], :, :]))
        src_label.append(np.squeeze(Label[ids[i], :]))

    fea_de = np.concatenate(src_data, axis=0)
    fea_de = data_normalize(fea_de, args.norm)
    fea_de = Variable(tr.from_numpy(fea_de).float())

    src_label = np.concatenate(src_label, axis=0)
    src_label = tr.from_numpy(src_label).long()
    print(fea_de.shape, src_label.shape)

    return fea_de, src_label


def read_seed_combine_tar(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = '/Users/wenz/dataset/MOABB/' + args.data + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wzw/data/bci/' + args.data + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    n_sub = len(Data_raw)
    ids = np.delete(np.arange(0, n_sub), args.idt)
    src_data, src_label = [], []
    for i in range(n_sub - 1):
        src_data.append(np.squeeze(Data_raw[ids[i], :, :]))
        src_label.append(np.squeeze(Label[ids[i], :]))
    src_data = np.concatenate(src_data, axis=0)
    src_label = np.concatenate(src_label, axis=0)

    src_data = data_normalize(src_data, args.norm)
    src_data = Variable(tr.from_numpy(src_data).float())
    src_label = tr.from_numpy(src_label).long()
    print(src_data.shape, src_label.shape)

    # target sub
    tar_data = np.squeeze(Data_raw[args.idt, :, :])
    tar_data = data_normalize(tar_data, args.norm)
    tar_data = Variable(tr.from_numpy(tar_data).float())
    tar_label = np.squeeze(Label[args.idt, :])
    tar_label = tr.from_numpy(tar_label).long()
    print(tar_data.shape, tar_label.shape)

    return src_data, src_label, tar_data, tar_label


def obtain_train_val_source(y_array, trial_ins_num, val_type):
    y_array = y_array.numpy()
    ins_num_all = len(y_array)
    src_idx = range(ins_num_all)

    if val_type == 'random':
        # 随机打乱会导致结果偏高，不管是MI还是SEED数据集
        num_train = int(0.9 * len(src_idx))
        id_train, id_val = tr.utils.data.random_split(src_idx, [num_train, len(src_idx) - num_train])

    if val_type == 'last':
        # 按顺序划分，一般情况来说没问题，但是如果源数据类别是按顺序排的，会有问题
        num_train = int(0.9 * trial_ins_num)
        id_train = np.array(src_idx).reshape(-1, trial_ins_num)[:, :num_train].reshape(1, -1).flatten()
        id_val = np.array(src_idx).reshape(-1, trial_ins_num)[:, num_train:].reshape(1, -1).flatten()

    return id_train, id_val
