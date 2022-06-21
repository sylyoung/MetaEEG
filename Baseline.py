import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from models.EEGNet import EEGNet
from models.ShallowConvNet import ShallowConvNet, ShallowConvNetReduced
from EEG_cross_subject_loader import EEG_loader

import random


def main(
        test_subj=None,
        learning_rate=None,
        num_iterations=None,
        cuda=None,
        seed=None,
        dataset=None,
        model_name=None,
        save=False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
        print('using cuda...')

    data = EEG_loader(test_subj=test_subj, dataset=dataset)
    train_x_arr = data.train_x
    train_y_arr = data.train_y
    train_x_arr_tmp = []
    train_y_arr_tmp = []

    for train_x, train_y in zip(train_x_arr, train_y_arr):
        train_x_arr_tmp.append(train_x)
        train_y_arr_tmp.append(train_y)

    tensor_train_x, tensor_train_y = torch.from_numpy(train_x).unsqueeze_(3).to(
        torch.float32), torch.squeeze(torch.from_numpy(train_y), 1).to(torch.long)

    train_dataset = TensorDataset(tensor_train_x, tensor_train_y)
    train_loader = DataLoader(train_dataset, batch_size=64)

    del data, train_x, train_y

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

    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if dataset == 'ERP1':
        class_weight = torch.tensor([1., 4.99], dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    elif dataset == 'ERP2':
        class_weight = torch.tensor([1., 2.42], dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(num_iterations):
        model.train()
        # print('epoch:', epoch + 1)
        total_loss = 0
        cnt = 0
        for i, (x, y) in enumerate(train_loader):
            # Forward pass
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss
            cnt += 1

            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
        out_loss = total_loss / cnt

        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_iterations, out_loss))

        if (epoch + 1) % 50 == 0 and epoch != 0 and save:
            # Save the model checkpoint
            torch.save(model, './runs/' + str(dataset) + '/invert_multi_' + model_name + dataset + '_seed' + str(
                seed) + '_pretrain_model_test_subj_' + str(test_subj) + '_epoch' + str(epoch + 1) + '.pt')
            # print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':

    model_name = 'ShallowConvNet'
    dataset = 'ERP2'

    if dataset == 'MI1':
        subj_num = 9
    elif dataset == 'MI2':
        subj_num = 14
    elif dataset == 'ERP1':
        subj_num = 10
    elif dataset == 'ERP2':
        subj_num = 16

    for test_subj in range(0, subj_num):
        for seed in range(0, 10):
            print('subj', test_subj, 'seed', seed)
            main(test_subj=test_subj,
                 learning_rate=0.001,
                 num_iterations=100,
                 cuda=False,
                 seed=seed,
                 dataset=dataset,
                 model_name=model_name,
                 save=True,
                 )
