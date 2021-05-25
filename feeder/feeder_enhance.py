# sys
import numpy as np
import pickle

# torch
import torch

# operation
from . import tools


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 window_stride=2,
                 debug=False,
                 enhance=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.enhance = enhance
        self.window_size = window_size
        self.window_stride = window_stride

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            label = pickle.load(f)

        # load data
        if mmap:
            data = np.load(self.data_path, mmap_mode='r')
        else:
            data = np.load(self.data_path)

        if self.debug:
            self.label = label[0:100]
            self.data = data[0:100]
            self.sample_name = self.sample_name[0:100]
        if self.enhance:
            # 数据增强
            self.data, self.label = tools.data_augmentation(data, label, self.window_size, self.window_stride)
            self.N, self.C, self.T, self.V, self.M = self.data.shape
        else:
            self.data, self.label = data, label
            self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        return data_numpy, label

