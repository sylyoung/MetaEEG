import learn2learn as l2l
import numpy as np
import torch

from models.EEGNet import EEGNet_features1, EEGNet_latter
from models.ShallowConvNet import ShallowConvNetFeatures, ShallowConvNetFeaturesReduced, ShallowConvNetClassifier
from EEG_cross_subject_loader import EEG_loader

import random


def main(
        test_subj=None,
        ways=None,
        shots=None,
        meta_lr=None,
        fast_lr=None,
        meta_batch_size=None,
        adaptation_steps=None,
        num_iterations=None,
        cuda=None,
        seed=None,
        model_name=None,
        dataset=None,
        se=None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:1')
        print('using cuda...')

    data = EEG_loader(test_subj=test_subj, dataset=dataset)
    train_x_arr = data.train_x
    train_y_arr = data.train_y
    train_x_arr_tmp = []
    train_y_arr_tmp = []
    for train_x, train_y in zip(train_x_arr, train_y_arr):
        train_x_arr_tmp.append(train_x)
        train_y_arr_tmp.append(train_y)

    l2l_train_tasks_arr = []

    for i in range(len(train_x_arr_tmp)):
        tensor_train_x, tensor_train_y = torch.from_numpy(train_x_arr_tmp[i]).unsqueeze_(3).to(
            torch.float32), torch.squeeze(torch.from_numpy(train_y_arr_tmp[i]), 1).to(torch.long)
        train_torch_dataset = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)
        train_loader = torch.utils.data.DataLoader(train_torch_dataset, batch_size=len(train_torch_dataset))
        l2l_train_tasks_arr.append(train_loader)
    del train_x_arr, train_y_arr, train_x_arr_tmp, train_y_arr_tmp, data, train_torch_dataset, tensor_train_x, tensor_train_y

    if model_name == 'ShallowConvNet':
        if dataset == 'MI1':
            model1 = ShallowConvNetFeatures(4, 22, 16640)
            model2 = ShallowConvNetClassifier(4, 22, 16640)
        if dataset == 'MI2':
            model1 = ShallowConvNetFeatures(2, 15, 26520)
            model2 = ShallowConvNetClassifier(2, 15, 26520)
        if dataset == 'ERP1':
            model1 = ShallowConvNetFeaturesReduced(2, 16, 6760)
            model2 = ShallowConvNetClassifier(2, 16, 6760)
        if dataset == 'ERP2':
            model1 = ShallowConvNetFeatures(2, 56, 17160)
            model2 = ShallowConvNetClassifier(2, 56, 17160)
    elif model_name == 'EEGNet':
        if dataset == 'MI1':
            model1 = EEGNet_features1(22, 256, 4)
            model2 = EEGNet_latter(22, 256, 4)
        if dataset == 'MI2':
            model1 = EEGNet_features1(15, 384, 2)
            model2 = EEGNet_latter(15, 384, 2)
        if dataset == 'ERP1':
            model1 = EEGNet_features1(16, 32, 2)
            model2 = EEGNet_latter(16, 32, 2)
        if dataset == 'ERP2':
            model1 = EEGNet_features1(56, 256, 2)
            model2 = EEGNet_latter(56, 256, 2)

    # subject number
    k = -1
    if dataset == 'MI1':
        k = 9
    if dataset == 'MI2':
        k = 14
    if dataset == 'ERP1':
        k = 10
    if dataset == 'ERP2':
        k = 16

    if dataset == 'ERP1':
        class_weight = torch.tensor([1., 4.99], dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    elif dataset == 'ERP2':
        class_weight = torch.tensor([1., 2.42], dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    former = model1
    latter = model2

    former.to(device)

    latter = l2l.algorithms.MAML(latter, lr=fast_lr, first_order=True, allow_nograd=True)
    latter.to(device)

    all_parameters = list(former.parameters()) + list(latter.parameters())
    opt = torch.optim.Adam(all_parameters, lr=meta_lr)
    loss = criterion

    print('start training...')
    for iteration in range(1, num_iterations + 1):

        opt.zero_grad()

        index_arr = []
        for i in range(len(l2l_train_tasks_arr)):
            target_domain_id = random.choice(np.arange(len(l2l_train_tasks_arr)))
            while i == target_domain_id:
                target_domain_id = random.choice(np.arange(len(l2l_train_tasks_arr)))
            index_arr.append(target_domain_id)
        print(index_arr)

        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        cnt = 0

        for i in range(len(l2l_train_tasks_arr)):

            train_task = next(iter(l2l_train_tasks_arr[i]))
            val_task = next(iter(l2l_train_tasks_arr[index_arr[i]]))

            val_data, val_labels = val_task
            val_data, val_labels = val_data.to(device), val_labels.to(device)
            former.eval()
            latter_cloned = latter.clone()
            x = former(val_data)
            valid_error = loss(latter_cloned(x), val_labels)

            latter_cloned = latter.clone()
            former.train()
            train_error, train_accuracy = fast_adapt(train_task,
                                                     val_task,
                                                     latter_cloned,
                                                     former,
                                                     loss,
                                                     adaptation_steps,
                                                     shots,
                                                     ways,
                                                     device)

            # backwardprop update if no negative transfer: target loss does not decrease
            if train_error < valid_error:
                train_error.backward()
                cnt += 1
                meta_train_error += train_error.item()
                meta_train_accuracy += train_accuracy.item()

        print('Iteration', iteration)

        if cnt == 0:
            print('no match this epoch')
            continue

        print('Meta Train Error', meta_train_error / cnt)
        print('Meta Train Accuracy', meta_train_accuracy / cnt)

        s = dataset + '_test_subj_' + str(test_subj) + '_shots_' + str(shots) + '_meta_lr_' + str(
            meta_lr) + '_fast_lr_' + \
            str(fast_lr) + '_meta_batch_size_' + str(meta_batch_size) + '_adaptation_steps_' + str(
            adaptation_steps) + str(model_name)

        if iteration % 50 == 0:
            print('saving model...')

            torch.save(former,
                       './runs/' + str(dataset) + '/mdmaml_model1_' + s + '_num_iterations_' + str(
                           iteration) + 'seed' + str(
                           se) + '.pt')

            torch.save(latter,
                       './runs/' + str(dataset) + '/mdmaml_model2_' + s + '_num_iterations_' + str(
                           iteration) + 'seed' + str(
                           se) + '.pt')

        # Average the accumulated gradients and optimize
        for p in all_parameters:
            if p.grad is None:
                continue
            p.grad.data.mul_(1.0 / cnt)
        opt.step()


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(adaptation_batch, evaluation_batch, latter, features1, loss, adaptation_steps, shots, ways, device):
    adaptation_data, adaptation_labels = adaptation_batch
    evaluation_data, evaluation_labels = evaluation_batch
    adaptation_data, adaptation_labels = adaptation_data.to(device), adaptation_labels.to(device)
    evaluation_data, evaluation_labels = evaluation_data.to(device), evaluation_labels.to(device)

    adaptation_data = features1(adaptation_data)

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(latter(adaptation_data), adaptation_labels)
        latter.adapt(train_error)

    evaluation_data = features1(evaluation_data)

    # Evaluate the adapted model
    predictions = latter(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)

    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


if __name__ == '__main__':

    meta_lr = 0.001
    fast_lr = 0.001
    for model_name in ['EEGNet', 'ShallowConvNet']:
        for dataset in ['MI1', 'MI2', 'ERP1', 'ERP2']:
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

            for test_subj in range(0, subj_num):
                for seed in range(0, 10):
                    print('MDMAML', dataset, model_name)
                    print('subj', test_subj, 'seed', seed)
                    main(test_subj=test_subj,
                         ways=ways,
                         shots=1,
                         meta_lr=meta_lr,
                         fast_lr=fast_lr,
                         meta_batch_size=1,
                         adaptation_steps=1,
                         num_iterations=500,
                         cuda=True,
                         seed=42,
                         model_name=model_name,
                         dataset=dataset,
                         se=seed,
                         )
