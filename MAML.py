import learn2learn as l2l
import numpy as np
import torch

from learn2learn.data.transforms import NWays, KShots, LoadData

from models.EEGNet import EEGNet
from models.ShallowConvNet import ShallowConvNet, ShallowConvNetReduced
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
        device = torch.device('cuda:4')
        print('using cuda...')

    data = EEG_loader(test_subj=test_subj, dataset=dataset)
    train_x_arr = data.train_x
    train_y_arr = data.train_y
    train_x_arr_tmp = []
    train_y_arr_tmp = []
    for train_x, train_y in zip(train_x_arr, train_y_arr):
        train_x_arr_tmp.append(train_x)
        train_y_arr_tmp.append(train_y)
    train_x_arr_tmp = np.concatenate(train_x_arr_tmp, axis=0)
    train_y_arr_tmp = np.concatenate(train_y_arr_tmp, axis=0)

    tensor_train_x, tensor_train_y = torch.from_numpy(train_x_arr_tmp).unsqueeze_(3).to(
        torch.float32), torch.squeeze(torch.from_numpy(train_y_arr_tmp), 1).to(torch.long)
    train_torch_dataset = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y)
    train_dataset = l2l.data.MetaDataset(train_torch_dataset)
    train_task = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=[
                                           NWays(train_dataset, n=ways),
                                           KShots(train_dataset, k=2 * shots),
                                           LoadData(train_dataset),
                                       ],
                                       num_tasks=meta_batch_size)
    del train_x_arr, train_y_arr, train_x_arr_tmp, train_y_arr_tmp , train_dataset, train_torch_dataset, tensor_train_x, tensor_train_y

    if model_name == 'ShallowConvNet':
        if dataset == 'MI1':
            model = ShallowConvNet(4, 22, 16640)
        if dataset == 'MI2':
            model = ShallowConvNet(2, 15, 26520)
        if dataset == 'ERP1':
            model = ShallowConvNetReduced(2, 16, 6760)
        if dataset == 'ERP2':
            model = ShallowConvNet(2, 56, 17160)
    elif model_name == 'EEGNet':
        if dataset == 'MI1':
            model = EEGNet(22, 256, 4)
        if dataset == 'MI2':
            model = EEGNet(15, 384, 2)
        if dataset == 'ERP1':
            model = EEGNet(16, 32, 2)
        if dataset == 'ERP2':
            model = EEGNet(56, 256, 2)

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

    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=True, allow_nograd=True)

    opt = torch.optim.Adam(maml.parameters(), lr=meta_lr)
    loss = torch.nn.CrossEntropyLoss()

    print('start training...')
    for iteration in range(1, num_iterations + 1):

        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0

        for batch in train_task:
                learner = maml.clone()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / (meta_batch_size))
        print('Meta Train Accuracy', meta_train_accuracy / (meta_batch_size))

        s = dataset + '_test_subj_' + str(test_subj) + '_shots_' + str(shots) + '_meta_lr_' + str(
            meta_lr) + '_fast_lr_' + \
            str(fast_lr) + '_meta_batch_size_' + str(meta_batch_size) + '_adaptation_steps_' + str(
            adaptation_steps) + str(model_name)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            if p.grad is None:
                continue
            p.grad.data.mul_(1.0 / (meta_batch_size))
        opt.step()

        if iteration % 50 == 0:
            print('saving model...')

            torch.save(model,
                       './runs/' + str(dataset) + '/maml_' + s + '_num_iterations_' + str(iteration) + 'seed' + str(se) + '.pth')


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


if __name__ == '__main__':

    meta_lr = 0.001
    fast_lr = 0.001
    shots = 10
    for model_name in ['EEGNet', 'ShallowConvNet']:
        for dataset in ['MI1', 'MI2', 'ERP1', 'ERP2']:
            if dataset == 'MI1':
                subj_num = 9
                meta_batch_size = 576 * 8 // (2 * 4 * shots)
            elif dataset == 'MI2':
                subj_num = 14
                meta_batch_size = 100 * 13 // (2 * 2 * shots)
            elif dataset == 'ERP1':
                subj_num = 10
                meta_batch_size = 575 * 9 // (2 * 2 * shots)
            elif dataset == 'ERP2':
                subj_num = 16
                meta_batch_size = 340 * 15 // (2 * 2 * shots)

            if dataset == 'MI1':
                ways = 4
            else:
                ways = 2

            for test_subj in range(0, subj_num):
                for seed in range(0, 10):
                    print('MAML', dataset, model_name)
                    print('subj', test_subj, 'seed', seed)
                    main(test_subj=test_subj,
                         ways=ways,
                         shots=shots,
                         meta_lr=meta_lr,
                         fast_lr=fast_lr,
                         meta_batch_size=meta_batch_size,
                         adaptation_steps=1,
                         num_iterations=200,
                         cuda=True,
                         seed=42,
                         model_name=model_name,
                         dataset=dataset,
                         se=seed,
                         )
