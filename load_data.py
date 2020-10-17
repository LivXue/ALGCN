from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
from util import BackgroundGenerator
import numpy as np


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CustomDataSet(Dataset):
    def __init__(
            self,
            images,
            texts,
            labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label

    def __len__(self):
        count = len(self.images)
        return count


def get_loader(path, batch_size):
    img_train = loadmat(path + "train_img.mat")['train_img']
    img_test = loadmat(path + "test_img.mat")['test_img']
    text_train = loadmat(path + "train_txt.mat")['train_txt']
    text_test = loadmat(path + "test_txt.mat")['test_txt']
    label_train = loadmat(path + "train_lab.mat")['train_lab']
    label_test = loadmat(path + "test_lab.mat")['test_lab']

    # missing labels
    # data_len = img_train.shape[0]
    # missing = int(data_len * 0.9)
    # label_train[0:missing] = 0

    # corrupting labels
    # noise_rates = [0.9, 0.9]
    # chances = np.random.uniform(size=label_train.shape)
    # condlist = [np.logical_and(label_train == 0, chances >= noise_rates[0]),
    #             np.logical_and(label_train == 0, chances < noise_rates[0]),
    #             np.logical_and(label_train == 1, chances >= noise_rates[1]),
    #             np.logical_and(label_train == 1, chances < noise_rates[1])]
    # choicelist = [0, 1, 1, 0]
    # label_train = np.select(condlist, choicelist)

    imgs = {'train': img_train, 'test': img_test}
    texts = {'train': text_train, 'test': text_test}
    labels = {'train': label_train, 'test': label_test}
    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['train', 'test']}

    shuffle = {'train': True, 'test': False}

    dataloader = {x: DataLoaderX(dataset[x], batch_size=batch_size,
                                 shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}

    img_dim = img_train.shape[1]
    text_dim = text_train.shape[1]
    num_class = label_train.shape[1]

    input_data_par = {}
    input_data_par['img_test'] = img_test
    input_data_par['text_test'] = text_test
    input_data_par['label_test'] = label_test
    input_data_par['img_train'] = img_train
    input_data_par['text_train'] = text_train
    input_data_par['label_train'] = label_train
    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    return dataloader, input_data_par
