# Neural Networks for Biometry on Body MRI

-Work in Progress-

This repository will contain code samples and documentation for regression with convolutional neural networks on medical images, described in the following publication: \
[_"Identifying morphological indicators of aging with neural networks on large-scale whole-body MRI"_](https://ieeexplore.ieee.org/document/8887538) [1]

Contents:
- PyTorch code for network models, training and prediction 
- Formatting of UK Biobank neck-to-knee body MRI (into volumes and 2D formats)
- Parameters for registration of UK Biobank neck-to-knee MRI with [2]

Please note that the UK Biobank data used in the publication can not be made publically available. However, the calculated reference values and split IDs used for the experiments have been shared as return data with the UK Biobank, so that reproducing the results should be possible. Feel free to contact us with any questions in this regard.

In some code samples new hyperparameters are used that have been found superior in subsequent experiments and will be described in upcoming work. It is still possible to run the code in the original configuration described in [1] if desired.


Contact: taro.langner@surgsci.uu.se

[1] [_T. Langner, J. Wikstrom, T. Bjerner, H. Ahlstrom, and J. Kullberg, “Identifying morphological indicators of aging with neural networks on large-scale whole-body MRI,” IEEE Transactions on Medical Imaging, pp. 1–1, 2019._](https://ieeexplore.ieee.org/document/8887538)\
[2] [_S. Ekström, F. Malmberg, H. Ahlström, J. Kullberg, and R. Strand, “Fast Graph-Cut Based Optimization for Practical Dense Deformable Registration of Volume Images,” arXiv:1810.08427 [cs], Oct. 2018. arXiv: 1810.08427_](https://arxiv.org/abs/1810.08427)
