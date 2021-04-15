import sys
import os

import time
import copy
import shutil

import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from torchvision import models

from models import net_vgg
from models import net_resnet

from dataset import VolumeDataset

# DISCLAIMER:
# This is not the exact code that was used to generate the results of the publications,
# but a condensed and cleaned, minimal version for convenience.

def main(argv):

    path_out = "cnn_project/" # output project folder

    path_gt = "../labelFormatting/liver_fat_all_vs_0/" # output of labelFormatting code, training set

    path_img = "/home/taro/DL_imagesets/UKB_fatwat_dual_proj32k_uint8/" # folder with .npy files

    B = 32 # batch size
    lr = 0.0001 # learning rate
    I = 6000 # Training iterations
    save_step = I / 6 # Iterations between snapshots
    arch = "resnet50" # network architecture
    gpu_id = 0

    use_standardization = True

    aug_t = np.array((0.0, 0.0, 16.0, 16.0)) # Augmentation, maximum translations: C, Z, Y, X
    lr_reduction_step = 5000 # Divide learning rate by 10 after this many iterations

    #####
    # OLD CONFIGURATION:
    # (Only use this to reproduce the original age estimation paper)
    #I = 80000 # Training iterations
    #lr_reduction_step = 99999 # Divide learning rate by 10 after this many iterations
    #arch = "vgg16" # network architecture
    #####

    ##
    start_time = time.time()
    train(path_out, path_gt, path_img, aug_t, B, lr, I, lr_reduction_step, save_step, arch, gpu_id, use_standardization)
    end_time = time.time()

    print("Elapsed training time: {}".format(end_time - start_time))


def train(path_out, path_gt, path_img, aug_t, B, lr, I, lr_reduction_step, save_step, arch, gpu_id, use_standardization):

    if not os.path.exists(path_out): os.makedirs(path_out)
    if not os.path.exists(path_out + "snapshots/"): os.makedirs(path_out + "snapshots/")

    (loader, N) = getDataLoader(path_gt, path_img, aug_t, B, path_out, use_standardization)

    print("Loading pretrained model...")

    if arch == "vgg16":
        net = getPretrainedVgg16(class_count=1, channel_count=2, dim=[256,256])
    elif arch == "resnet50":
        net = getPretrainedResNet50(class_count=1, channel_count=2, dim=[256,256])
    else:
        print("ERROR: Unknown architecture {}".format(arch))
        sys.exit()

    ##
    torch.backends.cudnn.benchmark = True
    net.train(True)
    net = net.cuda(gpu_id)

    optimizer = optim.Adam(net.parameters(), lr=lr)

    batch_count = N / float(B)
    E = int(np.ceil(I / batch_count))

    i = 0

    print("Training started...")
    for e in range(E):

        for X, Y in loader:
            
            # Skip rest after I iterations
            if i >= I: 
                continue

            X = X.cuda(gpu_id, non_blocking=True)
            Y = Y.cuda(gpu_id, non_blocking=True)

            optimizer.zero_grad()
            output = net(X)

            # Calculate loss
            loss = F.mse_loss(output, Y)

            loss.backward()
            optimizer.step()

            # Save checkpoints
            if i > 0 and ((i+1) % save_step) == 0:
            
                print("Storing snapshot {} of {}".format(int((i+1)/save_step), int(I / save_step)))

                state = {
                    'iteration': i+1,
                    'state_dict': net.state_dict(),
                    'optimizer' : optimizer.state_dict()}

                torch.save(state, path_out + "snapshots/snapshot_{}.pth.tar".format(i+1))

                # Reduce training rate by factor 10 for last sixth of training
                if i == lr_reduction_step:
                    print("    Lowering learning rate by factor 10")
                     
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10

            i += 1
            
    del net
    del loader


def getDataLoader(path_gt, path_img, aug_t, B, path_out, use_standardization):

    # Get data loader
    (dataset, slice_names) = getDataset(path_gt, path_img, aug_t, path_out, use_standardization)
    N = len(slice_names)

    # Data Loader
    params = {"batch_size": B,
              "shuffle":True,
              "num_workers": 8,
              "pin_memory": True,
              # use different random seeds for each worker
              # courtesy of https://github.com/xingyizhou/CenterNet/issues/233
              "worker_init_fn" : lambda id: np.random.seed(torch.initial_seed() // 2**32 + id) }

    loader = data.DataLoader(dataset, **params)

    return (loader, N)


def getDataset(path_gt, path_img, aug_t, path_out, use_standardization):

    # Get image files
    img_files = [f for f in os.listdir(path_img) if os.path.isfile(os.path.join(path_img, f))]
    img_files = [f for f in img_files if ".npy" in f]

    ids_img = [f.split(".")[0] for f in img_files]
    ids_img = np.array(ids_img).astype("int")

    # Get ground truth values
    with open(path_gt + "gt.txt") as f: entries = f.readlines()
    entries.pop(0)

    ids = [f.split(",")[0] for f in entries]
    values = [f.split(",")[1] for f in entries]

    ids = np.array(ids).astype("int")

    # Check if images exist for all ids from gt
    mask = np.invert(np.in1d(ids, ids_img))
    ids_missing = ids[mask]

    if len(ids_missing) > 0:
        print("ERROR: No image found for ids: {}".format(ids_missing))
        sys.exit()

    #
    img_paths = ["{}{}.npy".format(path_img, f) for f in ids]

    if use_standardization:
        values = np.array(values).astype("float")
        (values, mean, stdev) = standardizeValues(values)
        writeStandardization(path_out, mean, stdev)

    dataset = VolumeDataset(img_paths, values, aug_t)

    return (dataset, img_paths)


def standardizeValues(values):
 
     # Standardize
     mean = np.mean(values)
     values = values - np.mean(values)
 
     stdev = np.std(values, ddof=1)
     values = values / np.std(values, ddof=1)
 
     return (values, mean, stdev)


def writeStandardization(path_out, mean, stdev):
 
     with open(path_out + "standardization.txt", "w") as f:
         f.write("mean,{}\n".format(mean))
         f.write("stdev,{}\n".format(stdev))


def getPretrainedVgg16(class_count, channel_count, dim):

    net = net_vgg.Net(class_count, channel_count, dim)
    net_pretrained = models.vgg16_bn(pretrained=True)

    #
    net.features[0].weight.data = copy.copy(net_pretrained.features[0].weight.data[:, :channel_count, :, :])
    exclusions = ["features.0.weight", "features.0.bias", "classifier.0.weight", "classifier.0.bias", "classifier.6.weight", "classifier.6.bias"]

    copyParameters(net, net_pretrained, exclusions)

    del net_pretrained
    return net


def getPretrainedResNet50(class_count, channel_count, dim):

    net = net_resnet.Net(class_count, channel_count, dim)
    net_pretrained = models.resnet50(pretrained=True)

    #
    net.conv1.weight.data = copy.copy(net_pretrained.conv1.weight.data[:, :channel_count, :, :])
    exclusions = ["conv1.weight", "conv1.bias2", "fc.weight", "fc.bias"]

    copyParameters(net, net_pretrained, exclusions)

    del net_pretrained
    return net
    

def getPretrainedResNet50_backup(class_count, channel_count, dim):

    net = net_resnet.Net(class_count, channel_count, dim)
    net_pretrained = models.resnet50(pretrained=True)

    #
    net.conv1.weight.data = copy.copy(net_pretrained.conv1.weight.data[:, :channel_count, :, :])
    exclusions = ["conv1.weight", "conv1.bias2", "fc.weight", "fc.bias"]

    copyParameters(net, net_pretrained, exclusions)

    del net_pretrained
    return net


def copyParameters(net, net_pretrained, exclusions):

    params1 = net.named_parameters()
    params2 = net_pretrained.named_parameters()

    dict_params2 = dict(params2)

    #
    for name1, param1 in params1:
        if name1 not in exclusions:
            if name1 in dict_params2:
                param1.data.copy_(dict_params2[name1].data)


if __name__ == '__main__':
    main(sys.argv)
