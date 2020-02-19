This folder contains parameters for the co-alignment of MRI volumes with a deformable registration technique [1].
The technique itself is implemented [on its own GitHub page](https://github.com/simeks/deform). For saliency aggregation, 
it was used to align the masked fat fraction (FF) and water fraction (WF) volumes of a moving to a fixed subject as follows:


>./deform registration -f0 path_to_fixed_ff.nrrd -f1 /path_to_fixed_wf.nrrd -m0 path_to_moving_FF.nrrd -m1 path_to_moving_wf.nrrd -p params.txt -o output.vtk


[1] [_S. Ekström, F. Malmberg, H. Ahlström, J. Kullberg, and R. Strand, “Fast Graph-Cut Based Optimization for Practical Dense Deformable Registration of Volume Images,” arXiv:1810.08427 [cs], Oct. 2018. arXiv: 1810.08427_](https://arxiv.org/abs/1810.08427)
