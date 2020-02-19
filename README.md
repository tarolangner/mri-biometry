# Deep Regression for Biometry on Body MRI

This repository contains code samples and documentation for regression with convolutional neural networks on medical images, described in the following publications: \
[_"Identifying morphological indicators of aging with neural networks on large-scale whole-body MRI"_](https://ieeexplore.ieee.org/document/8887538) [1]
[_"Large-scale biometry with interpretable neural network regression on UK Biobank body MRI"_](https://ieeexplore.ieee.org/document/8887538) [2]

Contents:
- PyTorch code for network models, training and inference
- Formatting of UK Biobank neck-to-knee body MRI (into volumes and 2D formats)
- Parameters for registration of UK Biobank neck-to-knee MRI with [3]

The code contains the old network configuration [1] in comments, but by default uses the new, optimized hyperparameters and learning policy [2]. 
The saliency aggregation is currently not included. We used a modified [_GitHub repository by Utku Ozbulak_](https://github.com/utkuozbulak/pytorch-cnn-visualizations), which implements guided gradient-weighted class activation maps [4].

Please note that the UK Biobank data used in the publication can not be made publically available. However, the calculated reference values and split IDs used for the experiments have been shared as return data of application 14237 with the UK Biobank, so that reproducing the results should be possible. 

For any questions and suggestions, contact me at: taro.langner@surgsci.uu.se

[1] [_T. Langner, J. Wikstrom, T. Bjerner, H. Ahlstrom, and J. Kullberg, “Identifying morphological indicators of aging with neural networks on large-scale whole-body MRI,” IEEE Transactions on Medical Imaging, pp. 1–1, 2019._](https://ieeexplore.ieee.org/document/8887538)\
[2] [_T. Langner, H Ahlström, and J. Kullberg, “Large-scale biometry with interpretable neural network regression on UK Biobank body MRI,” arXiv:2002.06862 [cs], Feb. 2020. arXiv: 2002.06862_](https://arxiv.org/abs/2002.06862)\
[3] [_S. Ekström, F. Malmberg, H. Ahlström, J. Kullberg, and R. Strand, “Fast Graph-Cut Based Optimization for Practical Dense Deformable Registration of Volume Images,” arXiv:1810.08427 [cs], Oct. 2018. arXiv: 1810.08427_](https://arxiv.org/abs/1810.08427)\
[4] [_R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” in 2017 IEEE International Conference on Computer Vision (ICCV), (Venice), pp. 618–626, IEEE, Oct. 2017._](https://arxiv.org/abs/1610.02391)
