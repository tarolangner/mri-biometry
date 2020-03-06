import os
import sys
import io

import time

import zipfile
import pydicom

import numpy as np

import scipy.interpolate
import numba_interpolate

from skimage import filters

import nrrd
import cv2

c_out_pixel_spacing = np.array((2.23214293, 2.23214293, 3.))
c_resample_tolerance = 0.01 # Only interpolate voxels further off of the voxel grid than this

c_interpolate_seams = True # If yes, cut overlaps between stations to at most c_max_overlap and interpolate along them, otherwise cut at center of overlap
c_correct_intensity = True # If yes, apply intensity correction along overlap
c_max_overlap = 8 # Used in interpolation, any station overlaps are cut to be most this many voxels in size

c_trim_axial_slices = 4 # Trim this many axial slices from the output volume to remove folding artefacts

c_use_gpu = True # If yes, use numba for gpu access, otherwise use scipy on cpu

c_store_mip = True # If yes, extract 2d mean intensity projections as .npy
c_store_volumes = False # If yes, extract 3d volumes as .nrrd


def main(argv):

    input_file = "paths_input.txt"
    output_path = "output/"

    #ignore_errors = True
    ignore_errors = False

    if not os.path.exists(os.path.dirname(output_path)): os.makedirs(os.path.dirname(output_path))

    with open(input_file) as f: input_paths = f.read().splitlines()

    start_time = time.time()
    for i in range(len(input_paths)):

        dicom_path = input_paths[i]

        subject_id = os.path.basename(dicom_path).split("_")[0]
        output_file = output_path + "{}".format(subject_id)

        print("Processing subject {}: {}".format(i, subject_id))

        if ignore_errors:
            try:
                convertDicom(dicom_path, output_file)
            except:
                print("    Something went wrong with patient {}".format(subject_id))

        else:
            convertDicom(dicom_path, output_file)

    end_time = time.time()
    print("Elapsed time: {}".format(end_time - start_time))


##
# Extract mean intensity projection from input UK Biobank style DICOM zip
def convertDicom(input_path_zip, output_path):

    if not os.path.exists(input_path_zip):
        print("    ERROR: Could not find input file {}".format(input_path_zip))
        return

    # Get water and fat signal stations
    (voxels_w, voxels_f, positions, pixel_spacings) = getSignalStations(input_path_zip)

    origin = np.amin(np.array(positions), axis=0)

    # Resample stations onto output volume voxel grid
    (voxels_w, _, _, _)          = resampleStations(voxels_w, positions, pixel_spacings)
    (voxels_f, W, W_end, W_size) = resampleStations(voxels_f, positions, pixel_spacings)

    # Cut station overlaps to at most c_max_overlap
    (_, _, _, _, voxels_w)                 = trimStationOverlaps(W, W_end, W_size, voxels_w)
    (overlaps, W, W_end, W_size, voxels_f) = trimStationOverlaps(W, W_end, W_size, voxels_f)

    # Combine stations to volumes
    volume_w = fuseVolume(W, W_end, W_size, voxels_w, overlaps) 
    volume_f = fuseVolume(W, W_end, W_size, voxels_f, overlaps)

    # Create and store mean intensity projections
    storeOutput(volume_w, volume_f, output_path, origin)


def storeOutput(volume_w, volume_f, output_path, origin):

    if c_store_mip:

        mip_w = formatMip(volume_w)
        mip_f = formatMip(volume_f)

        #mip_out = np.dstack((mip_w, mip_f, np.zeros(mip_w.shape)))
        #cv2.imwrite(output_path + ".png", mip_out)

        mip_out = np.dstack((mip_w, mip_f)))
        np.save(output_path + ".npy", mip_out.transpose(2, 0, 1))

    if c_store_volumes:

        storeNrrd(volume_w, output_path + "_W", origin)
        storeNrrd(volume_f, output_path + "_F", origin)

        (volume_wf, volume_ff, volume_mask) = calculateFractions(volume_w, volume_f)

        storeNrrd(volume_wf, output_path + "_WF", origin)
        storeNrrd(volume_ff, output_path + "_FF", origin)
        storeNrrd(volume_mask, output_path + "_mask", origin)
        

def calculateFractions(volume_w, volume_f):

    volume_sum = volume_w + volume_f
    volume_sum[volume_sum == 0] = 1

    volume_wf = 1000 * volume_w / volume_sum
    volume_ff = 1000 * volume_f / volume_sum

    # Calculate body mask by getting otsu thresholds
    # for all coronal slices and applying their mean
    ts = np.zeros(volume_sum.shape[1])
    for i in range(volume_sum.shape[1]):
        ts[i] = filters.threshold_otsu(volume_sum[:, i, :])

    t = np.mean(ts)

    volume_mask = np.ones(volume_w.shape).astype("uint8")
    volume_mask[volume_sum < t] = 0

    return (volume_wf, volume_ff, volume_mask)


def storeNrrd(volume, output_path, origin):

    # See: http://teem.sourceforge.net/nrrd/format.html
    header = {'dimension': 3}
    header['type'] = "float"
    header['sizes'] = volume.shape

    # Spacing info compatible with 3D Slicer
    header['space dimension'] = 3
    header['space directions'] = np.array(c_out_pixel_spacing * np.eye(3,3))
    header['space origin'] = origin
    header['space units'] = "\"mm\" \"mm\" \"mm\""
    header['encoding'] = 'gzip'

    #
    nrrd.write(output_path + ".nrrd", volume, header, compression_level=1)


# Generate mean intensity projection 
def formatMip(volume):

    bed_width = 22
    volume = volume[:, :volume.shape[1]-bed_width, :]

    # Coronal projection
    slice_cor = np.sum(volume, axis = 1)
    slice_cor = np.rot90(slice_cor, 1)

    # Sagittal projection
    slice_sag = np.sum(volume, axis = 0)
    slice_sag = np.rot90(slice_sag, 1)

    # Normalize intensities
    slice_cor = (normalize(slice_cor) * 255).astype("uint8")
    slice_sag = (normalize(slice_sag) * 255).astype("uint8")

    # Combine to single output
    slice_out = np.concatenate((slice_cor, slice_sag), 1)
    slice_out = cv2.resize(slice_out, (256, 256))

    return slice_out


def normalize(img):

    img = img.astype("float")
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

    return img


def getSignalStations(input_path_zip):

    # Get stations from DICOM
    (stat_voxels, stat_names, stat_positions, stat_pixel_spacings, stat_timestamps) = stationsFromDicom(input_path_zip)

    # Find water and fat signal station data
    (voxels_w, positions_w, pixel_spacings, timestamps_w) = extractStationsForModality("_W", stat_names, stat_voxels, stat_positions, stat_pixel_spacings, stat_timestamps)
    (voxels_f, positions_f, _, timestamps_f)              = extractStationsForModality("_F", stat_names, stat_voxels, stat_positions, stat_pixel_spacings, stat_timestamps)

    # Ensure that water and fat stations match in position and size and non-redundant
    (stations_consistent, voxels_w, voxels_f, positions, pixel_spacings) = ensureStationConsistency(voxels_w, voxels_f, positions_w, positions_f, timestamps_w, timestamps_f, pixel_spacings)
    if not stations_consistent:
        print("    ERROR: Stations are inconsistent!")
        return

    return (voxels_w, voxels_f, positions, pixel_spacings)


def ensureStationConsistency(voxels_w, voxels_f, positions_w, positions_f, timestamps_w, timestamps_f, pixel_spacings):

    # Abort if water and fat stations are not in the same positions
    if not np.allclose(positions_w, positions_f):
        print("ABORT: Water and fat stations are not in the same position!")
        return (False, voxels_w, voxels_f, positions_w)

    (voxels_w, voxels_f, positions_w, positions_f, timestamps_w, timestamps_f, pixel_spacings) = removeDeprecatedStations(voxels_w, voxels_f, positions_w, positions_f, timestamps_w, timestamps_f, pixel_spacings)

    # Crop corresponding stations to same size where necessary
    for i in range(len(positions_w)):

        if not np.array_equal(voxels_w[i].shape, voxels_f[i].shape):

            print("WARNING: Corresponding stations {} have different dimensions: {} vs {} (Water vs Fat)".format(i, voxels_w[i].shape, voxels_f[i].shape))
            print("         Cutting to largest common size")

            # Cut to common size
            min_size = np.amin(np.vstack((voxels_w[i].shape, voxels_f[i].shape)), axis=0)

            voxels_w[i] = np.ascontiguousarray(voxels_w[i][:min_size[0], :min_size[1], :min_size[2]])
            voxels_f[i] = np.ascontiguousarray(voxels_f[i][:min_size[0], :min_size[1], :min_size[2]])

    # Sort by position
    pos_z = np.array(positions_w)[:, 2]
    (pos_z, pos_indices) = zip(*sorted(zip(pos_z, np.arange(len(pos_z))), reverse=True))

    voxels_w = [voxels_w[i] for i in pos_indices]
    positions_w = [positions_w[i] for i in pos_indices]
    timestamps_w = [timestamps_w[i] for i in pos_indices]

    voxels_f = [voxels_f[i] for i in pos_indices]
    positions_f = [positions_f[i] for i in pos_indices]
    timestamps_f = [timestamps_f[i] for i in pos_indices]

    pixel_spacings = [pixel_spacings[i] for i in pos_indices]

    return (True, voxels_w, voxels_f, positions_w, pixel_spacings)


def removeDeprecatedStations(voxels_w, voxels_f, positions_w, positions_f, timestamps_w, timestamps_f, pixel_spacings):

    # In case of redundant stations, choose the newest
    if len(np.unique(positions_w, axis=0)) != len(positions_w):

        seg_select = []

        for pos in np.unique(positions_w, axis=0):

            # Find stations at current position
            offsets = np.array(positions_w) - np.tile(pos, (len(positions_w), 1))
            dist = np.sum(np.abs(offsets), axis=1)

            indices_p = np.where(dist == 0)[0]

            if len(indices_p) > 1:

                # Choose newest station
                timestamps_w_p = [str(x).replace(".", "") for f, x in enumerate(timestamps_w) if f in indices_p]

                # If you get scanned around midnight its your own fault
                recent_p = np.argmax(np.array(timestamps_w_p))

                print("WARNING: Image stations ({}) are superimposed. Choosing most recently imaged one ({})".format(indices_p, indices_p[recent_p]))
                
                seg_select.append(indices_p[recent_p])
            else:
                seg_select.append(indices_p[0])
        
        voxels_w = [x for f,x in enumerate(voxels_w) if f in seg_select]        
        positions_w = [x for f,x in enumerate(positions_w) if f in seg_select]        
        timestamps_w = [x for f,x in enumerate(timestamps_w) if f in seg_select]        

        voxels_f = [x for f,x in enumerate(voxels_f) if f in seg_select]        
        positions_f = [x for f,x in enumerate(positions_f) if f in seg_select]        
        timestamps_f = [x for f,x in enumerate(timestamps_f) if f in seg_select]        

        pixel_spacings = [x for f,x in enumerate(pixel_spacings) if f in seg_select]        

    return (voxels_w, voxels_f, positions_w, positions_f, timestamps_w, timestamps_f, pixel_spacings)


def fuseVolume(W, W_end, W_size, voxels, overlaps):

    S = len(voxels)

    # Cast to datatype
    for i in range(S):  
        voxels[i] = voxels[i].astype("float32")

    # Taper off station edges linearly for later addition
    if c_interpolate_seams:
        voxels = fadeStationEdges(overlaps, W_size, voxels)

    # Adjust mean intensity of overlapping slices
    if c_correct_intensity:
        voxels = correctOverlapIntensity(overlaps, W_size, voxels)

    # Combine stations into volume by addition
    volume = combineStationsToVolume(W, W_end, voxels)

    # Remove slices affected by folding
    if c_trim_axial_slices > 0:
        start = c_trim_axial_slices
        end = volume.shape[2] - c_trim_axial_slices
        volume = volume[:, :, start:end]

    return volume


def combineStationsToVolume(W, W_end, voxels):

    S = len(voxels)

    volume_dim = np.amax(W_end, axis=0).astype("int")
    volume = np.zeros(volume_dim)

    for i in range(S):
        volume[W[i, 0]:W_end[i, 0], W[i, 1]:W_end[i, 1], W[i, 2]:W_end[i, 2]] += voxels[i][:, :, :]

    #
    volume = np.flip(volume, 2)
    volume = np.swapaxes(volume, 0, 1)

    return volume


def extractStationsForModality(tag, stat_names, stat_voxels, stat_positions, stat_pixel_spacings, stat_timestamps):

    # Merge all stats with given tag
    indices_t = [f for f, x in enumerate(stat_names) if str(tag) in str(x)]

    voxels_t = [x for f, x in enumerate(stat_voxels) if f in indices_t]
    positions_t = [x for f, x in enumerate(stat_positions) if f in indices_t]
    pixel_spacings_t = [x for f, x in enumerate(stat_pixel_spacings) if f in indices_t]
    timestamps_t = [x for f, x in enumerate(stat_timestamps) if f in indices_t]
    
    return (voxels_t, positions_t, pixel_spacings_t, timestamps_t)


def getSignalSliceNamesInZip(z):

    file_names = [f.filename for f in z.infolist()]

    # Search for manifest file (name may be misspelled)
    csv_name = [f for f in file_names if "manifest" in f][0]

    with z.open(csv_name) as f0:

        data = f0.read() # Decompress into memory

        entries = str(data).split("\\n")
        entries.pop(-1)

        # Remove trailing blank lines
        entries = [f for f in entries if f != ""]

        # Get indices of relevant columns
        header_elements = entries[0].split(",")
        column_filename = [f for f,x in enumerate(header_elements) if "filename" in x][0]

        # Search for tags such as "Dixon_noBH_F". The manifest header can not be relied on
        for e in entries:
            entry_parts = e.split(",")
            column_desc = [f for f,x in enumerate(entry_parts) if "Dixon_noBH_F" in x]

            if column_desc:
                column_desc = column_desc[0]
                break

        # Get slice descriptions and filenames
        descriptions = [f.split(",")[column_desc] for f in entries]
        filenames = [f.split(",")[column_filename] for f in entries]

        # Extract signal images only
        chosen_rows = [f for f,x in enumerate(descriptions) if "_W" in x or "_F" in x]
        chosen_filenames = [x for f,x in enumerate(filenames) if f in chosen_rows]

    return chosen_filenames


##
# Return, for S stations:
# R:     station start coordinates, shape Sx3
# R_end: station end coordinates,   shape Sx3
# dims:  station extents,           shape Sx3
# 
# Coordinates in R and R_end are in the voxel space of the first station
def getReadCoordinates(voxels, positions, pixel_spacings):

    S = len(voxels)

    # Convert from list to arrays
    positions = np.array(positions)
    pixel_spacings = np.array(pixel_spacings)

    # Get dimensions of stations
    dims = np.zeros((S, 3))
    for i in range(S):
        dims[i, :] = voxels[i].shape

    # Get station start coordinates
    R = positions
    origin = np.array(R[0])
    for i in range(S):
        R[i, :] = (R[i, :] - origin) / c_out_pixel_spacing

    R[:, 0] -= np.amin(R[:, 0])
    R[:, 1] -= np.amin(R[:, 1])
    R[:, 2] *= -1

    R[:, [0, 1]] = R[:, [1, 0]]

    # Get station end coordinates
    R_end = np.array(R)
    for i in range(S):
        R_end[i, :] += dims[i, :] * pixel_spacings[i, :] / c_out_pixel_spacing

    return (R, R_end, dims)


##
# Linearly taper off voxel values along overlap of two stations, 
# so that their addition leads to a linear interpolation.
def fadeStationEdges(overlaps, W_size, voxels):

    S = len(voxels)

    for i in range(S):

        # Only fade inwards facing edges for outer stations
        fadeToPrev = (i > 0)
        fadeToNext = (i < (S - 1))

        # Fade ending edge (facing to next station)
        if fadeToNext:

            for j in range(overlaps[i]):
                factor = (j+1) / (float(overlaps[i]) + 1) # exclude 0 and 1
                voxels[i][:, :, W_size[i, 2] - 1 - j] *= factor

        # Fade starting edge (facing to previous station)
        if fadeToPrev:

            for j in range(overlaps[i-1]):
                factor = (j+1) / (float(overlaps[i-1]) + 1) # exclude 0 and 1
                voxels[i][:, :, j] *= factor

    return voxels


## 
# Take mean intensity of slices at the edge of the overlap between stations i and (i+1)
# Adjust mean intensity of each slice along the overlap to linear gradient between these means
def correctOverlapIntensity(overlaps, W_size, voxels):

    S = len(voxels)

    for i in range(S - 1):
        overlap = overlaps[i]

        # Get average intensity at outer ends of overlap
        edge_a = voxels[i+1][:, :, overlap]
        edge_b = voxels[i][:, :, W_size[i, 2] - 1 - overlap]

        mean_a = np.mean(edge_a)
        mean_b = np.mean(edge_b)

        for j in range(overlap):

            # Get desired mean intensity along gradient
            factor = (j+1) / (float(overlap) + 1)
            target_mean = mean_b + (mean_a - mean_b) * factor

            # Get current mean of slice when both stations are summed
            slice_b = voxels[i][:, :, W_size[i, 2] - overlap + j]
            slice_a = voxels[i+1][:, :, j]

            slice_mean = np.mean(slice_a) + np.mean(slice_b)

            # Get correction factor
            correct = target_mean / slice_mean

            # correct intensity to match linear gradient
            voxels[i][:, :, W_size[i, 2] - overlap + j] *= correct
            voxels[i+1][:, :, j] *= correct

    return voxels


##
# Ensure that the stations i and (i + 1) overlap by at most c_max_overlap.
# Trim any excess symmetrically
# Update their extents in W and W_end
def trimStationOverlaps(W, W_end, W_size, voxels):

    W = np.array(W)
    W_end = np.array(W_end)
    W_size = np.array(W_size)

    S = len(voxels)
    overlaps = np.zeros(S).astype("int")

    for i in range(S - 1):
        # Get overlap between current and next station
        overlap = W_end[i, 2] - W[i + 1, 2]

        # No overlap
        if overlap <= 0:
            print("WARNING: No overlap between stations {} and {}. Image might be faulty.".format(i, i+1))

        # Small overlap which can for interpolation
        elif overlap <= c_max_overlap and c_interpolate_seams:
            print("WARNING: Overlap between stations {} and {} is only {}. Using this overlap for interpolation".format(i, i+1, overlap))

        # Large overlap which must be cut
        else:
            if c_interpolate_seams:
                # Keep an overlap of at most c_max_overlap
                cut_a = (overlap - c_max_overlap) / 2.
                overlap = c_max_overlap
            else:
                # Cut at center of seam
                cut_a = overlap / 2.
                overlap = 0

            cut_b = int(np.ceil(cut_a))
            cut_a = int(np.floor(cut_a))

            voxels[i] = voxels[i][:, :, 0:(W_size[i, 2] - cut_a)]
            voxels[i + 1] = voxels[i + 1][:, :, cut_b:]

            #
            W_end[i, 2] = W_end[i, 2] - cut_a
            W_size[i, 2] -= cut_a

            W[i + 1, 2] = W[i + 1, 2] + cut_b
            W_size[i + 1, 2] -= cut_b

        overlaps[i] = overlap

    return (overlaps, W, W_end, W_size, voxels)


##
# Station voxels are positioned at R to R_end, not necessarily aligned with output voxel grid
# Resample stations onto voxel grid of output volume
def resampleStations(voxels, positions, pixel_spacings):

    # R: station positions off grid respective to output volume
    # W: station positions on grid after resampling
    (R, R_end, dims) = getReadCoordinates(voxels, positions, pixel_spacings)

    # Get coordinates of voxels to write to
    W = np.around(R).astype("int")
    W_end = np.around(R_end).astype("int")
    W_size = W_end - W

    result_data = []

    #
    for i in range(len(voxels)):

        # Get largest offset off of voxel grid
        offsets = np.concatenate((R[i, :].flatten(), R_end[i, :].flatten()))
        offsets = np.abs(offsets - np.around(offsets))

        max_offset = np.amax(offsets)

        # Get difference in voxel counts
        voxel_count_out = np.around(W_size[i, :])
        voxel_count_dif = np.sum(voxel_count_out - dims[i, :])

        # No resampling if station voxels are already aligned with output voxel grid
        doResample = (max_offset > c_resample_tolerance or voxel_count_dif != 0)

        result = None
        
        if doResample:

            if c_use_gpu:

                # Use numba implementation on gpu:
                scalings = (R_end[i, :] - R[i, :]) / dims[i, :]
                offsets = R[i, :] - W[i, :] 
                result = numba_interpolate.interpolate3d(W_size[i, :], voxels[i], scalings, offsets)

            else:
                # Use scipy CPU implementation:
                # Define positions of station voxels (off of output volume grid)
                x_s = np.linspace(int(R[i, 0]), int(R_end[i, 0]), int(dims[i, 0]))
                y_s = np.linspace(int(R[i, 1]), int(R_end[i, 1]), int(dims[i, 1]))
                z_s = np.linspace(int(R[i, 2]), int(R_end[i, 2]), int(dims[i, 2]))

                # Define positions of output volume voxel grid
                y_v = np.linspace(W[i, 0], W_end[i, 0], W_size[i, 0])
                x_v = np.linspace(W[i, 1], W_end[i, 1], W_size[i, 1])
                z_v = np.linspace(W[i, 2], W_end[i, 2], W_size[i, 2])

                xx_v, yy_v, zz_v = np.meshgrid(x_v, y_v, z_v)

                pts = np.zeros((xx_v.size, 3))
                pts[:, 1] = xx_v.flatten()
                pts[:, 0] = yy_v.flatten()
                pts[:, 2] = zz_v.flatten()

                # Resample stations onto output voxel grid
                rgi = scipy.interpolate.RegularGridInterpolator((x_s, y_s, z_s), voxels[i], bounds_error=False, fill_value=None)
                result = rgi(pts)

        else:
            # No resampling necessary
            result = voxels[i]

        result_data.append(result.reshape(W_size[i, :]))

    return (result_data, W, W_end, W_size)


def groupSlicesToStations(sl_pixels, sl_series, sl_names, sl_positions, sl_pixel_spacings, sl_times):

    # Group by series into stats
    unique_series = np.unique(sl_series)

    #
    stat_voxels = []
    stat_series = []
    stat_names = []
    stat_positions = []
    stat_voxel_spacings = []
    stat_times = []

    # Each series forms one station
    for s in unique_series:

        # Get slice indices for series s
        indices_s = [f for f, x in enumerate(sl_series) if x == s]

        # Get physical positions of slice
        sl_positions_s = [x for f, x in enumerate(sl_positions) if f in indices_s]

        position_max = np.amax(np.array(sl_positions_s).astype("float"), axis=0)
        stat_positions.append(position_max)

        # Combine slices to stations
        voxels_s = slicesToStationData(indices_s, sl_positions_s, sl_pixels)
        stat_voxels.append(voxels_s)

        # Get index of first slice
        sl_0 = indices_s[0]

        stat_series.append(sl_series[sl_0])
        stat_names.append(sl_names[sl_0])
        stat_times.append(sl_times[sl_0])

        # Get 3d voxel spacing
        voxel_spacing_2d = sl_pixel_spacings[sl_0]

        # Get third dimension by dividing station extent by slice count
        z_min = np.amin(np.array(sl_positions_s)[:, 2].astype("float"))
        z_max = np.amax(np.array(sl_positions_s)[:, 2].astype("float"))
        z_spacing = (z_max - z_min) / (len(sl_positions_s) - 1)

        voxel_spacing = np.hstack((voxel_spacing_2d, z_spacing))
        stat_voxel_spacings.append(voxel_spacing)

    return (stat_voxels, stat_names, stat_positions, stat_voxel_spacings, stat_times)


def getDataFromDicom(ds):

    pixels = ds.pixel_array

    series = ds.get_item(["0020", "0011"]).value
    series = int(series)

    name = ds.get_item(["0008", "103e"]).value

    position = ds.get_item(["0020", "0032"]).value 
    position = np.array(position.decode().split("\\")).astype("float32")

    pixel_spacing = ds.get_item(["0028", "0030"]).value
    pixel_spacing = np.array(pixel_spacing.decode().split("\\")).astype("float32")

    start_time = ds.get_item(["0008", "0031"]).value

    return (pixels, series, name, position, pixel_spacing, start_time)


def slicesToStationData(slice_indices, slice_positions, slices):

    # Get size of output volume station
    slice_count = len(slice_indices)
    slice_shape = slices[slice_indices[0]].shape

    # Get slice positions
    slices_z = np.zeros(slice_count)
    for z in range(slice_count):
        slices_z[z] = slice_positions[z][2]

    # Sort slices by position
    (slices_z, slice_indices) = zip(*sorted(zip(slices_z, slice_indices), reverse=True))

    # Write slices to volume station
    dim = np.array((slice_shape[0], slice_shape[1], slice_count))
    station = np.zeros(dim)

    for z in range(dim[2]):
        slice_z_index = slice_indices[z]
        station[:, :, z] = slices[slice_z_index]

    return station


def stationsFromDicom(input_path_zip):

    # Get slice info
    pixels = []
    series = []
    names = []
    positions = []
    pixel_spacings = []
    times = []

    #
    z = zipfile.ZipFile(input_path_zip)

    signal_slice_names = getSignalSliceNamesInZip(z)

    for i in range(len(signal_slice_names)):

        # Read signal slices in memory
        with z.open(signal_slice_names[i]) as f0:

            data = f0.read() # Decompress into memory
            ds = pydicom.read_file(io.BytesIO(data)) # Read from byte stream

            (pixels_i, series_i, name_i, position_i, spacing_i, time_i) = getDataFromDicom(ds)

            pixels.append(pixels_i)
            series.append(series_i)
            names.append(name_i)
            positions.append(position_i)
            pixel_spacings.append(spacing_i)
            times.append(time_i)

    z.close()

    (stat_voxels, stat_names, stat_positions, stat_voxel_spacings, stat_times) = groupSlicesToStations(pixels, series, names, positions, pixel_spacings, times)

    return (stat_voxels, stat_names, stat_positions, stat_voxel_spacings, stat_times)


if __name__ == '__main__':
    main(sys.argv)
