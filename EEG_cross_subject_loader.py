import numpy as np
import scipy.io as sio


class EEG_loader():

    def __init__(self, test_subj=None, dataset=None):

        test_subj = test_subj
        data_folder = './data/' + str(dataset)

        train_x_arr = []
        train_y_arr = []

        prefix = 's'

        mat = sio.loadmat(data_folder + "/" + prefix + str(test_subj) + ".mat")
        x = np.moveaxis(np.array(mat['x']), -1, 0)
        y = np.array(mat['y'])
        test_x = x
        test_y = y

        a = 0
        if dataset == 'MI1':# 9 subjects
            k = 9
        elif dataset == 'MI2':# 14 subjects
            k = 14
        elif dataset == 'ERP1':# 10 subjects
            k = 10
        elif dataset == 'ERP2':# 16 subjects
            k = 16

        for i in range(a, k):

            mat = sio.loadmat(data_folder + "/" + prefix + str(i) + ".mat")
            x = np.moveaxis(np.array(mat['x']), -1, 0)
            y = np.array(mat['y'])

            train_x_arr.append(x)
            train_y_arr.append(y)

        train_x_array_out = []
        train_y_array_out = []
        for train_x, train_y in zip(train_x_arr, train_y_arr):

            np.random.seed(42)
            idx = list(range(len(train_y)))
            np.random.shuffle(idx)
            train_x = train_x[idx]
            train_y = train_y[idx]

            train_x_array_out.append(train_x)
            train_y_array_out.append(train_y)

        idx = list(range(len(test_y)))
        np.random.shuffle(idx)
        test_x = test_x[idx]
        test_y = test_y[idx]

        self.train_x = train_x_array_out
        self.train_y = train_y_array_out
        self.test_x = test_x
        self.test_y = test_y
