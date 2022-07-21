import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score

from models.EEGNet import EEGNet
from models.ShallowConvNet import ShallowConvNet, ShallowConvNetReduced
from EEG_cross_subject_loader import EEG_loader

import random
import sys
import time


def main(
        test_subj=None,
        learning_rate=None,
        adaptation_iterations=None,
        cuda=None,
        seed_num=None,
        shots=None,
        test_path=None,
        dataset=None,
        model_name=None,
        roc_auc=0,
):

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(1)
        device = torch.device('cuda:5')
        #print('using cuda...')


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

    if dataset == 'MI2':
        class_num = 4
    else:
        class_num = 2

    data = EEG_loader(test_subj=test_subj, dataset=dataset)

    if shots == 0:

        test_x, test_y = data.test_x, data.test_y
        tensor_test_x, tensor_test_y = torch.from_numpy(test_x).unsqueeze_(3).to(
            torch.float32), torch.squeeze(
            torch.from_numpy(test_y), 1).to(torch.long)

        test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
        test_loader = DataLoader(test_dataset)

        #model.load_state_dict(torch.load(test_path))
        model = torch.load(test_path)
        model.to(device)
        model.eval()

        y_true = []
        y_pred = []
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)

                y_true.append(y.item())
                y_pred.append(predicted.item())

            # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
            if roc_auc == '1':
                print('using roc_auc')
                out = roc_auc_score(y_true, y_pred)
            else:
                out = accuracy_score(y_true, y_pred)

        return round(out,3), 0

    accuracy_arr = []
    for seed in range(seed_num):
        target_x, target_y = data.test_x, data.test_y
        # print(target_x.shape, target_y.shape)
        # input('')

        np.random.seed(seed)
        idx = list(range(len(target_y)))
        np.random.shuffle(idx)
        target_x = target_x[idx]
        target_y = target_y[idx]

        calib_ind = np.ones(class_num * 2) * shots

        train_x = []
        train_y = []
        train_index = []
        for j in range(class_num):
            for i in range(len(target_y)):
                # print(target_y[i, 0])
                if target_y[i, 0] == j:
                    train_x.append(target_x[i])
                    train_y.append(target_y[i])
                    train_index.append([i])
                    calib_ind[j] -= 1
                    # print(calib_ind[j])
                    if calib_ind[j] == 0.0:
                        break
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        #print('calibration labels: ', train_y)
        # print(train_x.shape, train_y.shape)
        # print(train_index)

        test_x = np.delete(target_x, train_index, axis=0)
        test_y = np.delete(target_y, train_index, axis=0)
        # print(test_x.shape, test_y.shape)

        tensor_train_x, tensor_train_y = torch.from_numpy(train_x).unsqueeze_(3).to(
            torch.float32), torch.squeeze(
            torch.from_numpy(train_y), 1).to(torch.long)

        train_dataset = TensorDataset(tensor_train_x, tensor_train_y)
        train_loader = DataLoader(train_dataset)

        tensor_test_x, tensor_test_y = torch.from_numpy(test_x).unsqueeze_(3).to(
            torch.float32), torch.squeeze(
            torch.from_numpy(test_y), 1).to(torch.long)

        test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
        test_loader = DataLoader(test_dataset)

        # print(train_x.shape, test_x.shape)

        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        model.load_state_dict(torch.load(test_path))

        # Train the model
        for epoch in range(adaptation_iterations):
            # print('epoch:', epoch + 1)
            #total_loss = 0
            #cnt = 0
            for i, (x, y) in enumerate(train_loader):
                # Forward pass
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)
                #total_loss += loss
                #cnt += 1

                # Backward and optimize
                opt.zero_grad()
                loss.backward()
                opt.step()
            #out_loss = total_loss / cnt

            # print('Epoch [{}/{}], , Loss: {:.4f}'
            #      .format(epoch + 1, epoch, out_loss))

        # Test the model
        model.eval()

        y_true = []
        y_pred = []
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                y_true.append(y.item())
                y_pred.append(predicted.item())

            # print('Accuracy of the network on the test subject  : {} %'.format(100 * correct / total))
            if roc_auc == '1':
                print('using roc_auc')
                out = roc_auc_score(y_true, y_pred)
            else:
                out = accuracy_score(y_true, y_pred)
            accuracy_arr.append(round(out,3))
        #del data, target_x, target_y, tensor_train_x, tensor_train_y, tensor_test_x, tensor_test_y, train_dataset, test_dataset
    #print(accuracy_arr, round(np.average(accuracy_arr), 3), round(np.std(accuracy_arr), 3))
    #input('')
    return round(np.average(accuracy_arr), 3), round(np.std(accuracy_arr), 3)


if __name__ == '__main__':
    model_name = 'EEGNet'
    dataset = 'MI1'

    seed_num = 10 # test times of random calibration data
    learning_rate = 0.001 # normal learning rate
    meta_learning_rate = 0.001 # meta learning rate

    #shots = 1
    #adaptation_iterations = 2
    #test_load_epoch = 10

    mode = str(sys.argv[1])  # mode name
    shots = int(sys.argv[2])  # test subject calibration data of shots number (shots * classes)
    adaptation_iterations = int(sys.argv[3])  # test subject calibration data adaptation iterations/steps
    test_load_epoch = str(sys.argv[4])  # file of loaded saved parameters epoch number
    train_shots = str(sys.argv[5])  # file of loaded saved parameters shots number
    train_adaption_steps = str(sys.argv[6])  # file of loaded saved parameters adaptation steps number
    roc_auc = str(sys.argv[7])  # use roc_auc or not, 1 for yes, 0 for no

    avg_arr = []
    std_arr = []
    all_arr = []
    for i in range(0, 14):
        out_acc_arr = []
        out_std_arr = []
        for s in range(0, 1):
            print('subj', i, 'seed', s)
            if mode == 'base' or mode == 'finetune':
                # shots = 0
                path = './runs/' + str(dataset) + '/baseline_' + str(model_name) + str(dataset) + '_seed' + str(
                    s) + '_test_subj_' + str(
                    i) + '_epoch' + str(test_load_epoch) + '.pt'
            elif mode == 'maml':
                path = './runs/' + str(dataset) + '/' + 'maml_' + str(dataset) + '_test_subj_' + str(
                    i) + '_shots_' + str(
                    train_shots) + '_meta_lr_' + str(meta_learning_rate) + '_fast_lr_' + str(
                    learning_rate) + '_meta_batch_size_1_adaptation_steps_' + str(train_adaption_steps) + str(
                    model_name) + '_num_iterations_' + str(test_load_epoch) + 'seed' + str(s) + '.pth'
            elif mode == 'mdmaml':
                path = './runs/' + str(dataset) + '/' + 'mdmaml_' + str(dataset) + '_test_subj_' + str(
                    i) + '_shots_' + str(
                    train_shots) + '_meta_lr_' + str(meta_learning_rate) + '_fast_lr_' + str(
                    learning_rate) + '_meta_batch_size_1_adaptation_steps_' + str(train_adaption_steps) + str(
                    model_name) + '_num_iterations_'+ str(test_load_epoch)  + 'seed' + str(s) + '.pth'
                #'cdmaml_MI1_test_subj_1_shots_25_meta_lr_0.001_fast_lr_0.001_meta_batch_size_1_adaptation_steps_1EEGNetwithload_num_iterations_5seed0'
            elif mode == 'cdmaml+':
                path = './runs/' + str(dataset) + 'cdmaml+/' + 'cdmaml+_' + str(dataset) + '_test_subj_' + str(
                    i) + '_shots_' + str(
                    train_shots) + '_meta_lr_' + str(meta_learning_rate) + '_fast_lr_' + str(
                    learning_rate) + '_meta_batch_size_1_adaptation_steps_1' + str(
                    model_name) + '_num_iterations_' + str(test_load_epoch) + 'seed' + str(s) + '.pth'
            elif mode == 'cdmaml-':
                path = './runs/' + str(dataset) + 'cdmaml-/' + 'cdmaml-_' + str(dataset) + '_test_subj_' + str(
                    i) + '_shots_' + str(
                    train_shots) + '_meta_lr_' + str(meta_learning_rate) + '_fast_lr_' + str(
                    learning_rate) + '_meta_batch_size_1_adaptation_steps_1' + str(
                    model_name) + 'withload_num_iterations_' + str(test_load_epoch) + 'seed' + str(s) + '.pth'
            acc, std = main(
                test_subj=i,
                learning_rate=learning_rate,
                adaptation_iterations=adaptation_iterations,
                cuda=False,
                seed_num=seed_num,
                shots=shots,
                test_path=path,
                dataset=dataset,
                model_name=model_name,
                roc_auc=roc_auc,
            )
            print('score, std:', acc, std)
            out_acc_arr.append(round(acc, 3))
            out_std_arr.append(round(std, 3))
        print(out_acc_arr)
        all_arr.append(out_acc_arr)
        avg_arr.append(round(np.average(out_acc_arr), 3))
        std_arr.append(round(np.std(out_acc_arr), 5))

    total_avg = round(np.average(avg_arr), 5)
    total_std = round(np.std(np.average(all_arr, axis=0)), 5)
    print('#' * 32)
    print(dataset, model_name, mode, '\nshots:', shots, '; iter:', adaptation_iterations, '; loaded_epoch:',
          test_load_epoch)
    print('avg_arr:', avg_arr)
    print('std_arr:', std_arr)
    print('total_avg:', total_avg)
    print('total_std:', total_std)
