import os
import sys

import torch
from torch.utils import data
from torch.autograd import Variable
import numpy as np

import scipy
from scipy import ndimage

class VolumeDataset(data.Dataset):

    def __init__(self, img_paths, labels, aug_t):
        self.labels = labels
        self.img_paths = img_paths
        self.aug_t = aug_t

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        # Select sample
        img_path = self.img_paths[index]

        # Load data and get label
        img = np.load(img_path)

        D = len(img.shape)

        if np.count_nonzero(self.aug_t) > 0:

            disp = np.around((2 * np.random.random(D) - 1) * self.aug_t[-D:])
            img = scipy.ndimage.shift(img, disp, order=0, mode="nearest")

        X = Variable(torch.from_numpy(img)).float()
        Y = Variable(torch.from_numpy(np.array(float(self.labels[index])))).float()

        return X, Y
