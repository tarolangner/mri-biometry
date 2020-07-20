# Convert DICOMs
Extract fused 3d volumes (_.nrrd_), 2d mean intensity projections (_.npy_) or 2d fat fraction slices containing liver tissue from UK Biobank neck-to-knee body MRI DICOM data (field 20201).

Only the 2d formats are used as input to the networks in this repository.

By settings c\_store\_volumes to _True_, the fused 3d volumes can be extracted, with the water signal, fat signal, water fraction, fat fraction and body mask. These volumes can be co-aligned by deformable registration.

The reported speed was reached by storing all DICOMs on an internal or external SSD, and using a GPU for image resampling.

# Interpolation
The six overlapping MRI stations must be fused into one volume. By default, the overlap is cropped to (c_max_overlap=8) transverse slices and interpolated with intensity correction. The interpolation was not applied to the volumes used in the saliency aggregation. Its purpose is to smooth out motion artefacts between the stations and the intensity correction takes into account signal loss along the outer slices. This interpolation is mostly cosmetic and likely not a requirement for accurate results with the deep regression approach.
