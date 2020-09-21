# Liver fat inference
This directory contains a ResNet50 snapshot trained for inference of liver fat. It corresponds to the instance evaluated on dataset B of the following manuscript: \
[_"Large-scale inference of liver fat with neural networks on UK Biobank body MRI"_](https://arxiv.org/abs/2006.16777)


The snapshot can be found at: *snapshots/resnet50_liver_fat_inference.pth.tar*

The two-dimensional fat fraction slices can be extracted with *../dicomToProjection/convertDicoms* by activating the flag *c_store_ff_slice*.
