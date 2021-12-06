import numpy as np
from scipy.io import loadmat, savemat

if __name__ == '__main__':
    dataset = 'mirflickr'
    DATA_DIR = 'data/' + dataset + '/'
    adj_file = 'data/' + dataset + '/adj.mat'
    label_train = loadmat(DATA_DIR + "train_lab.mat")['train_lab']

    num_class = label_train.shape[1]
    adj = np.zeros((num_class, num_class), dtype=int)
    num = np.zeros((num_class), dtype=int)

    for row in label_train:
        for i in range(num_class):
            if row[i] == 0:
                continue
            else:
                num[i] += 1

            for j in range(i, num_class):
                if row[j] == 1:
                    adj[i][j] += 1
                    adj[j][i] += 1

    file = {'adj': adj, 'nums': num}
    savemat(adj_file, file)
