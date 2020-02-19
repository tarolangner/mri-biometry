import sys
import os
import numpy as np

from dataset import VolumeDataset
from torch.utils import data

import torch

def getDataset(data_path, subsets, aug_t):

    # Get path to image files
    image_path_file = data_path + "imageset_path.txt"
    with open(image_path_file) as f:
        entries = f.read().splitlines()
    images_path = entries[0]

    img_names = []
    img_labels = []

    # Define img_paths
    for k in subsets:
        
        # Load image names for subset
        img_names_file_k = data_path + "subsets/subset_{}_ids.txt".format(k)
        with open(img_names_file_k) as f:
            img_names_k = f.read().splitlines()

        img_label_file_k = data_path + "subsets/subset_{}_labels.txt".format(k)
        with open(img_label_file_k) as f:
            img_labels_k = f.read().splitlines()

        img_names.extend(img_names_k)
        img_labels.extend(img_labels_k)

    img_paths = [images_path + f + ".npy" for f in img_names]
    img_labels = np.array(img_labels)

    dataset = VolumeDataset(img_paths, img_labels, aug_t)

    return (dataset, img_names)


def inferImageMetrics(data_path):

    image_path_file = data_path + "imageset_path.txt"
    with open(image_path_file) as f:
        entries = f.read().splitlines()

    images_path = entries[0]

    files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

    # Load one file
    file_name = files[0]
    image = np.load(images_path + file_name)

    dim = image.shape[1:]
    channel_count = image.shape[0]

    return (channel_count, dim)
