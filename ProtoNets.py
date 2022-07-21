import learn2learn as l2l
import numpy as np
import torch

from learn2learn.data.transforms import NWays, KShots, LoadData
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

from models.EEGNet import EEGNet_features
from models.ShallowConvNet import ShallowConvNetFeatures, ShallowConvNetFeaturesReduced
from EEG_cross_subject_loader import EEG_loader

import random


def main(
        test_subj=None,
        ways=None,
        shots=None,
        lr=None,
        num_iterations=None,
        cuda=None,
        seed=None,
        model_name=None,
        dataset=None,
        se=None,
        test=True,
        path=None,
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:4')
        print('using cuda...')

    data = EEG_loader(test_subj=test_subj, dataset=dataset)
    if not test:
        train_x_arr = data.train_x
        train_y_arr = data.train_y
        train_x_arr_tmp = []
        train_y_arr_tmp = []
        for train_x, train_y in zip(train_x_arr, train_y_arr):
            train_x_arr_tmp.append(train_x)
            train_y_arr_tmp.append(train_y)

        l2l_train_tasks_arr = []
        meta_batch_size = 8
        for i in range(len(train_x_arr_tmp)):
            tensor_train_x, tensor_train_y = torch.from_numpy(train_x_arr_tmp[i]).unsqueeze_(3).to(
                torch.float32), torch.squeeze(torch.from_numpy(train_y_arr_tmp[i]), 1).to(torch.long)
            train_torch_dataset = TensorDataset(tensor_train_x, tensor_train_y)
            train_dataset = l2l.data.MetaDataset(train_torch_dataset)
            train_tasks = l2l.data.TaskDataset(train_dataset,
                                               task_transforms=[
                                                   NWays(train_dataset, n=ways),
                                                   KShots(train_dataset, k=2 * shots),
                                                   LoadData(train_dataset),
                                               ],
                                               num_tasks=meta_batch_size)
            l2l_train_tasks_arr.append(train_tasks)
        del train_x_arr, train_y_arr, train_x_arr_tmp, train_y_arr_tmp, data, train_dataset, train_torch_dataset, tensor_train_x, tensor_train_y
    else:
        test_x, test_y = data.test_x, data.test_y

        tensor_test_x, tensor_test_y = torch.from_numpy(test_x).unsqueeze_(3).to(
            torch.float32), torch.squeeze(
            torch.from_numpy(test_y), 1).to(torch.long)

        test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)

    if model_name == 'ShallowConvNet':
        if dataset == 'MI1':
            model = ShallowConvNetFeatures(4, 22, 16640)
        if dataset == 'MI2':
            model = ShallowConvNetFeatures(2, 15, 26520)
        if dataset == 'ERP1':
            model = ShallowConvNetFeaturesReduced(2, 16, 6760)
        if dataset == 'ERP2':
            model = ShallowConvNetFeatures(2, 56, 17160)
    elif model_name == 'EEGNet':
        if dataset == 'MI1':
            model = EEGNet_features(22, 256, 4)
        if dataset == 'MI2':
            model = EEGNet_features(15, 384, 2)
        if dataset == 'ERP1':
            model = EEGNet_features(16, 32, 2)
        if dataset == 'ERP2':
            model = EEGNet_features(56, 256, 2)

    # subject number
    k = -1
    if dataset == 'MI2':
        k = 9
    if dataset == 'MI1':
        k = 14
    if dataset == 'ERP2':
        k = 16
    if dataset == 'ERP1':
        k = 10

    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=20, gamma=0.5)

    if not test:
        print('start training...')
        for iteration in range(1, num_iterations + 1):

            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            cnt = 0

            for tasks_subj_ind in range(len(l2l_train_tasks_arr)):
                for batch in l2l_train_tasks_arr[tasks_subj_ind]:
                    loss, acc = fast_adapt(model,
                                           batch,
                                           ways,
                                           shots,
                                           metric=pairwise_distances_logits,
                                           device=device)

                    cnt += 1
                    meta_train_error += loss.item()
                    meta_train_accuracy += acc.item()

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            lr_scheduler.step()

            print('Iteration', iteration)
            print('Meta Train Error', meta_train_error / cnt)
            print('Meta Train Accuracy', meta_train_accuracy / cnt)

            s = dataset + '_test_subj_' + str(test_subj) + '_' + str(model_name)

            if iteration % 50 == 0:
                print('saving model...')

                torch.save(model,
                           './runs/' + str(dataset) + '/protonets_' + s + '_num_iterations_' + str(iteration) +
                           '_seed' + str(se) + '.pt')

    else:
        model = torch.load(path)
        model.eval()

        metric_arr = []
        seed_num = 10
        for seed in range(seed_num):

            pred_arr = []
            targ_arr = []
            for i, batch in enumerate(test_loader, 1):
                predictions, targets = fast_adapt_test(model,
                                       batch,
                                       ways,
                                       shots,
                                       metric=pairwise_distances_logits,
                                       device=device)

                predictions, targets = predictions.tolist(), targets.tolist()

                pred_arr.extend(predictions)
                targ_arr.extend(targets)

            if dataset == 'ERP1' or dataset == 'ERP2':
                score = roc_auc_score(targets, predictions)
            else:
                score = accuracy_score(targets, predictions)
            metric_arr.append(round(score,5))
        print(metric_arr)
        print(round(np.average(metric_arr),5))
        print(round(np.std(metric_arr), 5))


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def auc_counter(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return predictions.numpy(), targets.numpy()


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1)) ** 2).sum(dim=2)
    return logits


def fast_adapt(model, batch, ways, shot, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)

    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot * 2)
    for offset in range(shot):
        if (selection + offset)[-1] >= len(support_indices):
            return 0,0
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


def fast_adapt_test(model, batch, ways, shot, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot * 2)
    for offset in range(shot):
        if (selection + offset)[-1] >= len(support_indices):
            return 0,0
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = metric(query, support)
    #loss = F.cross_entropy(logits, labels)
    predictions, targets = auc_counter(logits, labels)

    return predictions, targets


if __name__ == '__main__':

    lr = 0.001
    num_iterations = 100
    test = False
    for model_name in ['EEGNet', 'ShallowConvNet']:
        for dataset in ['MI1', 'MI2', 'ERP1', 'ERP2']:
            if dataset == 'MI1':
                subj_num = 9
                shots = 576 // (2 * 4 * 8) + 1
                if test:
                    shots = 2 # 4 6
            elif dataset == 'MI2':
                subj_num = 14
                shots = 100 // (2 * 2 * 8) + 1
                if test:
                    shots = 2 # 4 6
            elif dataset == 'ERP1':
                subj_num = 10
                shots = 96 // (2 * 8) + 1
                if test:
                    shots = 2 # 4 6
            elif dataset == 'ERP2':
                subj_num = 16
                shots = (24 + 26) // (2 * 8) + 1 # pad to 50
                if test:
                    shots = 2 # 4 6

            if dataset == 'MI1':
                ways = 4
            else:
                ways = 2

            test_load_epoch = 100
            for subj in range(0, subj_num):
                for se in range(0, 10):
                    path = './runs/' + str(dataset) + '/protonet_' + str(dataset) + '_test_subj_' + str(subj) + '_' + \
                           str(model_name) + '_num_iterations_' + str(test_load_epoch) + '_seed' + str(se) + '.pth'
                    print('ProtoNet', dataset, model_name)
                    print('subj', subj, 'seed', se)
                    main(test_subj=subj,
                         ways=ways,
                         shots=shots,
                         lr=lr,
                         num_iterations=num_iterations,
                         cuda=True,
                         seed=42,
                         model_name=model_name,
                         dataset=dataset,
                         se=se,
                         test=test,
                         path=path,
                         )
