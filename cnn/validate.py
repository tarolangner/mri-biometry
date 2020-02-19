import sys
import os
import re

import time
import copy

import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from models import net_vgg
from models import net_resnet

from dataset import VolumeDataset

from sklearn.metrics import r2_score


def main(argv):

    path_in = "cnn_project/"

    path_gt = "../labelFormatting/liver_fat_cv0/" # output of labelFormatting code, validation set
    path_img = "/home/taro/DL_imagesets/UKB_fatwat_dual_proj32k_uint8/"

    B = 32
    gpu_id = 0

    #arch = "vgg16"
    arch = "resnet50"

    if not os.path.exists(path_in + "eval/"): os.makedirs(path_in + "eval/")

    predict(path_in, path_gt, path_img, B, arch, gpu_id)


def predict(path_in, path_gt, path_img, B, arch, gpu_id):

    (loader, img_files) = getDataLoader(path_gt, path_img, B, path_in)

    if arch == "vgg16":
        net = net_vgg.Net(num_classes=1, num_channels=2, dim=[256, 256])
    elif arch == "resnet50":
        net = net_resnet.Net(num_classes=1, num_channels=2, dim=[256, 256])
    else:
        print("ERROR: Unknown architecture {}".format(arch))
        sys.exit()

    # Find snapshots
    path_snapshots = path_in + "snapshots/"
    files = [f for f in natural_sort(os.listdir(path_snapshots)) if os.path.isfile(os.path.join(path_snapshots, f))]
    files = [f for f in files if ".pth.tar" in f]

    S = len(files)
    N = len(img_files)

    # 
    values_gt = np.zeros((S, N))
    values_out = np.zeros((S, N))

    # Get predictions by snapshot
    for i in range(S):
        print("Evaluating {}...".format(files[i]))
        (values_gt[i, :], values_out[i, :]) = predictWithSnapshot(path_snapshots + files[i], net, loader, N, B, gpu_id)

    #
    path_std = path_in + "standardization.txt"
    if os.path.exists(path_std):
        (std_mean, std_stdev) = readStandardization(path_std)
        values_gt = values_gt * std_stdev + std_mean
        values_out = values_out * std_stdev + std_mean

    #
    print("Writing output to {}".format(path_in + "eval/"))
    writePredictions(path_in, files, img_files, values_gt, values_out)
    writeMetrics(path_in, values_gt, values_out, files)


def writeMetrics(path_in, values_gt, values_out, files):

    S = values_gt.shape[0]

    with open(path_in + "/eval/evaluation.txt", "w") as f:

        f.write("Snapshot,MAE,R^2,r\n")

        for i in range(S):
            r = np.corrcoef(np.transpose(np.stack((values_gt[i, :], values_out[i, :]), axis=1)))[1, 0]
            r2 = r2_score(values_gt[i, :], values_out[i, :])
            mae = np.mean(np.abs(values_gt[i, :] - values_out[i, :]))

            f.write("{},{},{},{}\n".format(files[i], mae, r2, r))


def writePredictions(path_in, files, img_files, values_gt, values_out):

    for s in range(len(files)):
        
        it = files[s].split(".")[0].split("_")[1]
        path_eval = path_in + "/eval/predictions_{}.txt".format(it)

        with open(path_eval, "w") as f: 
            f.write("img,gt,out\n")

            for i in range(len(values_gt[s])):
                f.write("{},{},{}\n".format(img_files[i], values_gt[s, i], values_out[s, i]))
            

def readStandardization(path_std):

    if not os.path.exists(path_std):
        print("ERROR: No standardization file found. Did labelFormatting/formatLabel run with useStandardization = True?")
        sys.exit()

    # Revert standardization
    with open(path_std) as f:
        entries = f.readlines()

    #
    std_mean = float(entries[0].split(",")[1].replace("\n",""))
    std_stdev = float(entries[1].split(",")[1].replace("\n",""))

    return (std_mean, std_stdev)


def predictWithSnapshot(path_snapshot, net, loader, N, B, gpu_id):

    snapshot = torch.load(path_snapshot, map_location={"cuda" : "cpu"})
    net.load_state_dict(snapshot['state_dict'])

    #
    torch.backends.cudnn.benchmark = True
    net = net.cuda(gpu_id)
    net.eval()

    values_gt = np.zeros(N)
    values_out = np.zeros(N)
    
    i_start = 0

    for X, Y in loader:

        X = X.cuda(gpu_id, non_blocking=True)

        out = net(X).cpu().data[:].numpy()

        B_i = B
        i_end = i_start + B_i

        # Last batch may wrap over, so only use unique entries
        if i_end > N:
            B_i = N % B
            i_end = i_start + B_i

        values_out[i_start:i_end] = out[:B_i]
        values_gt[i_start:i_end] = Y[:B_i]

        i_start = i_end

    return (values_gt, values_out)


def getDataLoader(path_gt, path_img, B, path_in):

    # Get data loader
    (dataset, img_files) = getDataset(path_gt, path_img, path_in)

    # Data Loader
    params = {"batch_size": B,
              "shuffle":False,
              "num_workers": 8,
              "pin_memory": True}

    loader = data.DataLoader(dataset, **params)

    return (loader, img_files)


def getDataset(path_gt, path_img, path_in):

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

    path_std = path_in + "standardization.txt"
    if os.path.exists(path_std):
        print("Applying label standardization...")
        values = np.array(values).astype("float")
        (std_mean, std_stdev) = readStandardization(path_std)
        values = (values - std_mean) / std_stdev

    aug_t = np.array((0, 0, 0, 0))
    dataset = VolumeDataset(img_paths, values, aug_t)

    return (dataset, img_paths)


def natural_sort(l): 

    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 

    return sorted(l, key = alphanum_key)


if __name__ == '__main__':
    main(sys.argv)
