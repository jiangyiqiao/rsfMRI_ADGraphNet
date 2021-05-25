# sys
import os
from scipy.io import loadmat
import numpy as np
import json
# torch
import torch

# operation
from . import tools


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        random_move: If true, perform randomly but continuously changed transformation to input sequence
        window_size: The length of the output sequence
        pose_matching: If ture, match the pose between two frames
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 pose_matching=False,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.pose_matching = pose_matching
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        # load file list
        datas = loadmat(self.data_path)
        self.corr = datas['data']
        self.label = datas['label']
        self.sample_nums = len(self.corr)

        # load label
        self.label = np.array([0 if self.label[i] == 'NC  ' else 1 if self.label[i] == 'EMCI' else 2 for i in range(self.sample_nums)], dtype=np.int32)

        # output data shape (N, C, T, V, M)
        self.N = self.sample_nums  #sample
        self.C = 90  #channel
        self.T = 130  #frame
        self.V = 90  #joint
        self.M = 1  #person

    def __len__(self):
        return self.sample_nums

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        # fill data_numpy
        data_numpy = np.zeros((self.C, self.T, self.V, 1))
        for frame_index in range(130):
            corr = self.corr[index][0]
            data_numpy[:, frame_index, :, 0] = corr[:, :, frame_index]

        # get & check label index
        label = self.label[index]

        # data augmentation
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        # print(data_numpy.shape, type(data_numpy))
        # data_numpy = data_numpy.transpose((1, 2, 0, 3))

        return data_numpy, label

    def top_k(self, score, top_k):
        assert (all(self.label >= 0))

        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def top_k_by_category(self, score, top_k):
        assert (all(self.label >= 0))
        return tools.top_k_by_category(self.label, score, top_k)

    def calculate_recall_precision(self, score):
        assert (all(self.label >= 0))
        return tools.calculate_recall_precision(self.label, score)