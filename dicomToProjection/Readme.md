Extract fused 3d volumes (_.nrrd_) and 2d mean intensity projections (_.npy_) from UK Biobank neck-to-knee body MRI DICOM data (field 20201).

Only the 2d mean intensity projections are needed as input to the network.

By settings c\_store\_volumes to _True_, the fused 3d volumes can be extracted, with the water signal, fat signal, water fraction, fat fraction and body mask. These volumes can be co-aligned by deformable registration.

The reported speed was reached by storing all DICOMs on an internal or external SSD, and using a GPU for image resampling.
