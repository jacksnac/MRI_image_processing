import subprocess, os, sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import subprocess

from scipy import ndimage

# from scipy.spatial import distance
# from sklearn import cluster
# from sklearn.metrics import mutual_info_score


###############################################
######### Processing a single ROI #############
###############################################

DEBUG = False
DTYPE = np.int

#
# Select the struct to use
#
def roi_struct(option = 2):

    # the 26 neighbor system
    struct_26 = np.zeros((3, 3, 3))

    # 18 neighbors, take 8 corners out
    struct_18 = np.zeros((3, 3, 3))
    struct_18[:, 1, :] = 1
    struct_18[1, :, :] = 1
    struct_18[:, :, 1] = 1

    # 6 neighbors, only keep the 6 facet centers 
    struct_6 = np.zeros((3, 3, 3))
    struct_6[1, 1, 0] = 1
    struct_6[1, 1, 2] = 1
    struct_6[0, 1, 1] = 1
    struct_6[2, 1, 1] = 1
    struct_6[1, 0, 1] = 1
    struct_6[1, 2, 1] = 1

    # 26 neighbors
    if option == 1:
        return struct_26

    # 18 neighbors, take 8 corners out
    elif option == 2:   
        return struct_18     

    # 6 neighbors, only keep the 6 facet centers 
    elif option == 3:
        return struct_6
    # return 18 neighbor if option is not specified
    else:
        return struct_18

#
# Dilation
#
def roi_dilate(roi_fln, option = 2):


    struct = roi_struct(option)
    print('INFO: the kernel ', struct)

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')

    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header
   
    opt_dat_bin = np.zeros_like(roi_dat)
    opt_dat_bin[np.where(roi_dat > 0)] = 1.0
    opt_dat = ndimage.binary_dilation(opt_dat_bin, structure = struct).astype(DTYPE)
    opt_fln = roi_fln + '_dilate.nii.gz'

    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    nib.save(opt_img, opt_fln)
    
    return 1         


#
# Erosion
#
def roi_erode(roi_fln, option = 2):


    struct = roi_struct(option)
    print('INFO: the kernel ', struct)

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')


    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header
   
    opt_dat_bin = np.zeros_like(roi_dat)
    opt_dat_bin[np.where(roi_dat > 0)] = 1.0
    opt_dat = ndimage.binary_erosion(opt_dat_bin, structure = struct).astype(DTYPE)
    opt_fln = roi_fln + '_erode.nii.gz'

    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    nib.save(opt_img, opt_fln)
    
    return 1     

#
# Opening
#
def roi_opening(roi_fln, option = 2):

    struct = roi_struct(option)
    print('INFO: the kernel ', struct)

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')


    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header
   
    opt_dat_bin = np.zeros_like(roi_dat)
    opt_dat_bin[np.where(roi_dat > 0)] = 1.0
    opt_dat = ndimage.binary_opening(opt_dat_bin, structure = struct).astype(DTYPE)
    opt_fln = roi_fln + '_opening.nii.gz'

    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    nib.save(opt_img, opt_fln)
    
    return 1   


#
# Closing
#
def roi_closing(roi_fln, option = 2):


    struct = roi_struct(option)
    print('INFO: the kernel ', struct)

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .niiinstead')
        roi_img = nib.load(roi_fln + '.nii')

    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header
   
    opt_dat_bin = np.zeros_like(roi_dat)
    opt_dat_bin[np.where(roi_dat > 0)] = 1.0
    opt_dat = ndimage.binary_closing(opt_dat_bin, structure = struct).astype(DTYPE)
    opt_fln = roi_fln + '_closing.nii.gz'

    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    nib.save(opt_img, opt_fln)
    
    return 1   



#
# Construct the ROI core, rim, and shells
#
def roi_rings(roi_fln, ref_fln, num_shl = 3, option = 2):

    struct = roi_struct(option)
    print('INFO: the kernel ', struct)

    try:      
        ref_img = nib.load(ref_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the REF image in NII.GZ. Trying to load the .NII instead.')
        ref_img = nib.load(ref_fln + '.nii')
    ref_dat = ref_img.get_data()
    ref_aff = ref_img.affine
    ref_hdr = ref_img.header

    try:
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    # Flip the images if the affine transformation matrix is different
    if roi_aff[0, 0] * ref_aff[0, 0] < 0:
        roi_dat = roi_dat[::-1, :, :]
        print('INFO: Flip left and right.')
        
    if roi_aff[1, 1] * ref_aff[1, 1] < 0:
        roi_dat = roi_dat[:,::-1, :]
        print('INFO: Flip anterior and posterior.')
        
    if roi_aff[2, 2] * ref_aff[2, 2] < 0:
        roi_dat = roi_dat[:, :, ::-1]
        print('INFO: Flip superior and inferior.')

    # The tempory data arrays
    tmp_dat_seq = []
    # The outputs file names
    opt_fln_seq = []

    if not (roi_img.shape == ref_img.shape):
        print('ERROR: image dimension not match!!!')
        print('INFO: Lesion mask ', roi_img.shape, roi_img.get_data_dtype())
        print('INFO: Reference ', ref_img.shape, ref_img.get_data_dtype())
    else:            
        # Define the binary lesion image
        opt_dat_bin = roi_dat
        opt_dat_bin[np.where(roi_dat > 0)] = 1.0            
        # Define the lesion core images
        opt_dat_core = ndimage.binary_erosion(opt_dat_bin, structure = struct).astype(DTYPE)         
        # Add the core and binary array to the tmp array
        opt_fln_seq.append(roi_fln + '_core.nii.gz')
        tmp_dat_seq.append(opt_dat_core)
        opt_fln_seq.append(roi_fln + '_bin.nii.gz')
        tmp_dat_seq.append(opt_dat_bin)
        opt_fln_seq.append(roi_fln + '_rim.nii.gz')                  
        # Define the lesion ring images
        tmp_dat_pre = opt_dat_bin
        for i in np.arange(num_shl) + 1:
            opt_fln_seq.append(roi_fln + '_shell_' + str(i) + '.nii.gz')
            tmp_dat = ndimage.binary_dilation(tmp_dat_pre, structure = struct).astype(DTYPE)
            tmp_dat_seq.append(tmp_dat)
            tmp_dat_pre = tmp_dat

    # Save the image
    for j in np.arange(len(opt_fln_seq)):
        # Save the _core, _binary files
        if j in [0, 1]:
            opt_img = nib.Nifti1Image(tmp_dat_seq[j], ref_aff, header = ref_hdr)     
        # Save the _ring* files
        else:
            tmp_dat_sub = np.copy(tmp_dat_seq[j - 1])
            tmp_dat_sub[np.where(tmp_dat_seq[j - 2] > 0)] = 0.
            opt_img = nib.Nifti1Image(tmp_dat_sub, ref_aff, header = ref_hdr)                   
        nib.save(opt_img, opt_fln_seq[j])

    return 1



#
# Construct the ROI core, rim, and shells
#
def roi_shell_n(roi_fln, num_shl = 3, option = 2):

    struct = roi_struct(np.int(option))
    print('INFO: the kernel ', struct)

    try:
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header
     
    # Define the binary lesion image
    opt_dat = np.zeros_like(roi_dat)
    opt_dat[np.where(roi_dat > 0)] = 1  
    # Define the lesion core images

    for i in np.arange(np.int(num_shl)) + 1:
        opt_dat = ndimage.binary_dilation(opt_dat, structure = struct)
    
    opt_dat[np.where(roi_dat > 0)] = 0    
    opt_fln = roi_fln + '_shell_n_' + str(i*100 + np.int(option)) + '.nii.gz'
    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    nib.save(opt_img, opt_fln)

    return 1



#
# Construct the ROI core, rim, and shells
#
def roi_shell(roi_fln, num_shl = 3, option = 2):

    struct = roi_struct(np.int(option))
    print('INFO: the kernel ', struct)

    try:
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header
     
    # Define the binary lesion image
    opt_dat = np.zeros_like(roi_dat)
    opt_dat[np.where(roi_dat > 0)] = 1.0            
    # Define the lesion core images

    for i in np.arange(np.int(num_shl)) + 1:
        tmp_dat = opt_dat
        opt_dat = ndimage.binary_dilation(tmp_dat, structure = struct).astype(DTYPE)
    
    opt_dat[np.where(tmp_dat > 0)] = 0.    
    opt_fln = roi_fln + '_shell_' + str(i*100 + np.int(option)) + '.nii.gz'
    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    nib.save(opt_img, opt_fln)

    return 1




#
# Dilate the ROI multiple times
#
def roi_dilate_n(roi_fln, num_shl = 3, option = 2):

    struct = roi_struct(np.int(option))
    print('INFO: the kernel ', struct)

    try:
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header
     
    # Define the binary lesion image
    opt_dat = np.zeros_like(roi_dat)
    opt_dat[np.where(roi_dat > 0)] = 1.0            
    # Define the lesion core images

    for i in np.arange(np.int(num_shl)) + 1:
        opt_dat = ndimage.binary_dilation(opt_dat, structure = struct).astype(DTYPE)
    
    opt_fln = roi_fln + '_dilate_' + str(i*100 + np.int(option)) + '.nii.gz'
    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    nib.save(opt_img, opt_fln)

    return 1

#
# Flip to match a reference
#
def roi_flip(roi_fln, ref_fln):

    try:      
        ref_img = nib.load(ref_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the REF image in NII.GZ. Trying to load the .NII instead.')
        ref_img = nib.load(ref_fln + '.nii')
    ref_dat = ref_img.get_data()
    ref_aff = ref_img.affine
    ref_hdr = ref_img.header
    
    try:
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    opt_dat = roi_dat
    opt_fln_ext = '_flip.nii.gz'
    # Flip the images if the affine transformation matrix is different
    if roi_aff[0, 0] * ref_aff[0, 0] < 0:
        opt_dat = roi_dat[::-1, :, :]
        print('INFO: Flip left and right.')
        
    if roi_aff[1, 1] * ref_aff[1, 1] < 0:
        opt_dat = roi_dat[:,::-1, :]
        print('INFO: Flip anterior and posterior.')
        
    if roi_aff[2, 2] * ref_aff[2, 2] < 0:
        opt_dat = roi_dat[:, :, ::-1]
        print('INFO: Flip superior and inferior.')    


    if not (roi_img.shape == ref_img.shape):
        print('ERROR: image dimension not match!!!')
        print('INFO: Lesion mask ', roi_img.shape, roi_img.get_data_dtype())
        print('INFO: Reference ', ref_img.shape, ref_img.get_data_dtype())
        return -1
    else:
        dx0, dx1, dx2 = roi_img.shape
        print('INFO: Dimensions of the images ', dx0, dx1, )
        opt_img = nib.Nifti1Image(opt_dat, ref_aff, header = ref_hdr)               
        nib.save(opt_img, roi_fln + opt_fln_ext)
        return 1    



#
# Split one ROI into multiple ROIs based on the voxel indices
#
def roi_split(roi_fln):

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data().astype(int)
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    for i in np.arange(np.max(roi_dat)) + 1:
        opt_dat = np.zeros_like(roi_dat)
        opt_dat[np.where(roi_dat == i)] = i
        opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
        nib.save(opt_img, roi_fln + '_split_' + str(i) + '.nii.gz')

    return 1



#
# Extract one single value from a multiple-valued ROI
#
def roi_extract(roi_fln, val):

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data().astype(int)
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    v = np.int(val)

    if v == 0:
        v = np.max(roi_dat[:])

    if v < 0:
        u = np.unique(roi_dat)
        v = u[v]

    opt_dat = np.zeros_like(roi_dat)
    opt_dat[np.where(roi_dat == v)] = v
    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    nib.save(opt_img, roi_fln + '_extract_' + str(v) + '.nii.gz')

    return 1




#
# Binarize the ROI image
#
def roi_bin(roi_fln):

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    opt_dat = np.zeros_like(roi_dat)
    opt_dat[np.where(roi_dat > 0)] = 1.
    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    nib.save(opt_img, roi_fln + '_bin.nii.gz')

    return 1



#
# Remove certain value from the ROI
#

def roi_prescind(roi_fln, val, opt_fln):

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data().astype(np.int16)
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    opt_dat = roi_dat
    opt_dat[np.where(roi_dat == np.int(val))] = 0
    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)

    if opt_fln is None:
        opt_fln = roi_fln + '_precind_'+ str(val)

    nib.save(opt_img, opt_fln + '.nii.gz')

    return 1



#
# Binarize the ROI images by choosing a threshold
#
def roi_thr(roi_fln, min, max, opt_fln):

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    opt_dat = np.zeros_like(roi_dat)
    roi_coords = (roi_dat >= np.float(min)) & (roi_dat <= np.float(max))
    # opt_dat[roi_coords] = roi_dat[roi_coords]
    opt_dat[roi_coords] = 1.0
    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)

    if opt_fln is None:
        opt_fln = roi_fln + '_thr_'+ str(min) + '_' + str(max)

    nib.save(opt_img, opt_fln + '.nii.gz')

    return 1


#
# Calculate the center and extend of a ROI
#
def roi_extend(roi_fln):

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data().astype(int)
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    num_roi = np.max(roi_dat)
    # number of variables
    num_var = 6

    opt_arr = np.ndarray((num_roi, num_var))

    for i in np.arange(num_roi) + 1:

        roi_coords = np.where(roi_dat == i)

        if not roi_coords is None: 
            
            x_min = roi_coords[0].min().astype(float)
            x_max = roi_coords[0].max().astype(float)
            y_min = roi_coords[1].min().astype(float)
            y_max = roi_coords[1].max().astype(float)
            z_min = roi_coords[2].min().astype(float)
            z_max = roi_coords[2].max().astype(float)

            center_x = (x_min + x_max)/2
            center_y = (y_min + y_max)/2
            center_z = (z_min + z_max)/2
            radius_x = (x_max - x_min + 1)/2
            radius_y = (y_max - y_min + 1)/2
            radius_z = (z_max - z_min + 1)/2

            opt_vec = [center_x, center_y, center_z, radius_x, radius_y, radius_z]

            opt_arr[i-1, :] = opt_vec[:]

            print(i, opt_vec)

    return opt_arr


# 
# Crop an image based on an ROI
# 
def roi_crop(ipt_fln, roi_fln):

    try:      
        ipt_img = nib.load(ipt_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        ipt_img = nib.load(ipt_fln + '.nii')
    ipt_dat = ipt_img.get_data()
    ipt_aff = ipt_img.affine
    ipt_hdr = ipt_img.header


    try:
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    if not (ipt_img.shape == roi_img.shape):
        print('ERROR: image dimension not match!!!')
        print('INFO: input volume ', ipt_img.shape, ipt_img.get_data_dtype())
        print('INFO: ROI mask ', roi_img.shape, roi_img.get_data_dtype())
        return -1

    else:
        roi_pos = roi_dat > 0
        opt_dat = np.zeros_like(ipt_dat)
        opt_dat[roi_pos] = ipt_dat[roi_pos]

        opt_fln = ipt_fln + '_crop.nii.gz'
        opt_img = nib.Nifti1Image(opt_dat, ipt_aff, header = ipt_hdr)
        nib.save(opt_img, opt_fln)




#
# Blur an ROI using Gaussian Filter
#

def roi_gaussian(ipt_fln, kernel_size = 0.4):

    from skimage.filters import gaussian

    try:      
        ipt_img = nib.load(ipt_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        ipt_img = nib.load(ipt_fln + '.nii')

    ipt_dat = ipt_img.get_data()
    ipt_aff = ipt_img.affine
    ipt_hdr = ipt_img.header

    opt_dat = gaussian(ipt_dat, sigma = np.float(kernel_size))
    opt_fln = ipt_fln + '_gaussian.nii.gz'
    opt_img = nib.Nifti1Image(opt_dat, ipt_aff, header = ipt_hdr)
    nib.save(opt_img, opt_fln)


###############################################
#### Processing based on ROI and Reference ####
###############################################


#
# Calculate the statistics of the voxels given a ROI mask
#
def roi_stats(ref_fln, roi_fln):

    try:      
        ref_img = nib.load(ref_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the REF image in NII.GZ. Trying to load the .NII instead.')
        ref_img = nib.load(ref_fln + '.nii')
    ref_dat = ref_img.get_data()
    ref_aff = ref_img.affine
    ref_hdr = ref_img.header
    
    try:
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data().astype(int)
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    num_roi = np.max(roi_dat)
    # number of statistics
    num_sta = 6

    opt_arr = np.ndarray((num_roi, num_sta))

    print('Index: [Volumn, Min, Max, Mean, Median, STD]')

    for i in np.arange(num_roi) + 1:

        roi_values = ref_dat[roi_dat == i]

        # if not roi_values is None:
        if roi_values.any():  
            roi_vol = roi_values.size
            roi_min = roi_values.min()
            roi_max = roi_values.max()
            roi_mean = roi_values.mean()
            roi_median = np.median(roi_values)
            roi_std = roi_values.std()

            opt_vec = [roi_vol, roi_min, roi_max, roi_mean, roi_median, roi_std]

            opt_arr[i-1, :] = opt_vec[:]

            print(i, opt_vec)
            # print roi_vol

    return opt_arr

#
# Select topN from the lesion
#
def roi_topn(roi_fln, ref_fln, ord = 0, topN = 50):

    try:      
        ref_img = nib.load(ref_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the REF image in NII.GZ. Trying to load the .NII instead')
        ref_img = nib.load(ref_fln + '.nii')
    ref_dat = ref_img.get_data()
    ref_aff = ref_img.affine
    ref_hdr = ref_img.header

    if ord == 1:
        ref_dat = ref_dat.max() - ref_dat
        print('INFO: Selecting hypointensive / bright voxels')
    else:
        print('INFO: Selecting hyperintensive / dark voxels')


    try:
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header


    if not (roi_img.shape == ref_img.shape):
        print('ERROR: image dimension not match!!!')
        print('INFO: Lesion mask ', roi_img.shape, roi_img.get_data_dtype())
        print('INFO: Reference ', ref_img.shape, ref_img.get_data_dtype())
        return -1
    else:
        opt_dat = np.zeros_like(roi_dat)
        opt_fln = roi_fln + '_top_' + str(topN) + '.nii.gz'

        # Process the ROIs individually
        for i in np.arange(np.max(roi_dat)) + 1:
            roi_tmp_coords = np.where(roi_dat == i)
            roi_tmp_values = ref_dat[roi_tmp_coords] 
            
            if topN > roi_tmp_values.shape[0]: topN = roi_tmp_values.shape[0]
            
            ind_topN = roi_tmp_values.argsort()[0:topN]    
            roi_tmp_values_topN = roi_tmp_values[ind_topN]            
            # find out where these voxels are
            roi_tmp_coords_topN = [roi_tmp_coords[0][ind_topN], roi_tmp_coords[1][ind_topN], roi_tmp_coords[2][ind_topN]]
            
            #  to validate the results, validate the values in the filtered mask, 
            #  the topN values in the original mask, and the values in the manual labelled mask
            if DEBUG:          
                roi_tmp_values_filtered = ref_dat[roi_tmp_coords_topN]
                print('INFO: top ', topN, ' values in the original mask')
                print(roi_tmp_values_topN)
                print('INFO: top ', topN, ' values in the filtered mask')
                print(roi_tmp_values_filtered)  
            
            roi_nod_set = set()
            for j in np.arange(topN):
                ind_current = ind_topN[j]
                roi_nod_set.add((roi_tmp_coords[0][ind_current], roi_tmp_coords[1][ind_current], roi_tmp_coords[2][ind_current]))
            if DEBUG: print('INFO: Input nod list', roi_nod_set)

            min_score = roi_tmp_values.min()
            min_index = roi_tmp_values.argmin()    

            sed_nod = (roi_tmp_coords[0][min_index], roi_tmp_coords[1][min_index], roi_tmp_coords[2][min_index])
            opt_nod_set, COUNTER = search_neighbor(roi_nod_set, sed_nod, set(), 0)

            for nod in opt_nod_set:
                opt_dat[nod] = i    
        
        opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
        nib.save(opt_img, opt_fln)
    
        return 1  



#
# Remove the partial volume effect
#
def roi_pve(roi_fln, pve_fln, opt_fln):

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header


    try:
        pve_img = nib.load(pve_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        pve_img = nib.load(pve_fln + '.nii')
    pve_dat = pve_img.get_data()
    pve_aff = pve_img.affine
    pve_hdr = pve_img.header

    if not (roi_img.shape == pve_img.shape):
        print('ERROR: image dimension not match!!!')
        print('INFO: Lesion mask ', roi_img.shape, pve_img.get_data_dtype())
        print('INFO: Reference ', pve_img.shape, pve_img.get_data_dtype())
        return -1

    else:
        opt_dat = np.zeros_like(roi_dat)
        opt_dat[np.where(roi_dat > 0)] = 1.
        opt_dat[np.where(pve_dat > 0)] = 0.

    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)

    if opt_fln is None: 
        opt_fln = roi_fln + '_pve.nii.gz'
    else: 
        opt_fln = opt_fln + '.nii.gz'

    nib.save(opt_img, opt_fln)
    
    return 1     



#
# Intersect with another ROI
#
def roi_intersect(roi1_fln, roi2_fln, opt_fln):

    try:      
        roi1_img = nib.load(roi1_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi1_img = nib.load(roi1_fln + '.nii')
    roi1_dat = roi1_img.get_data()
    roi1_aff = roi1_img.affine
    roi1_hdr = roi1_img.header


    try:
        roi2_img = nib.load(roi2_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi2_img = nib.load(roi2_fln + '.nii')
    roi2_dat = roi2_img.get_data()
    roi2_aff = roi2_img.affine
    roi2_hdr = roi2_img.header

    if not (roi1_img.shape == roi2_img.shape):
        print('ERROR: image dimension not match!!!')
        print('INFO: 1st lesion mask ', roi1_img.shape, roi1_img.get_data_dtype())
        print('INFO: 2nd lesion mask ', roi2_img.shape, roi2_img.get_data_dtype())
        return -1

    else:
        opt_dat = np.zeros_like(roi1_dat)
        opt_dat[(roi1_dat > 0) & (roi2_dat > 0)] = 1.

    opt_img = nib.Nifti1Image(opt_dat, roi1_aff, header = roi1_hdr)

    if opt_fln is None: 
        opt_fln = roi_fln + '.nii.gz'
    else: 
        opt_fln = opt_fln + '.nii.gz'

    nib.save(opt_img, opt_fln)
    
    return 1    




#
# Intersect a set of iamges
#

def roi_intersect_n(roi_fln_list, opt_fln):

    try: 
        roi1_fln = roi_fln_list[0]
        roi2_fln = roi_fln_list[1]
        roi_intersect(roi1_fln, roi2_fln, opt_fln)
        for roi_fln in roi_fln_list[2:]:
            roi_intersect(opt_fln, roi_fln, opt_fln)
    except FileNotFoundError:
        print('ERROR: Cannot load images. ')



#
# Merge two ROI images
#
def roi_merge(roi1_fln, roi2_fln, opt_fln):

    try:      
        roi1_img = nib.load(roi1_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi1_img = nib.load(roi1_fln + '.nii')
    roi1_dat = roi1_img.get_data()
    roi1_aff = roi1_img.affine
    roi1_hdr = roi1_img.header


    try:
        roi2_img = nib.load(roi2_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi2_img = nib.load(roi2_fln + '.nii')
    roi2_dat = roi2_img.get_data()
    roi2_aff = roi2_img.affine
    roi2_hdr = roi2_img.header

    if not (roi1_img.shape == roi2_img.shape):
        print('ERROR: image dimension not match!!!')
        print('INFO: 1st lesion mask ', roi1_img.shape, roi1_img.get_data_dtype())
        print('INFO: 2nd lesion mask ', roi2_img.shape, roi2_img.get_data_dtype())
        return -1

    else:
        opt_dat = np.zeros_like(roi1_dat)
        opt_dat[np.where(roi1_dat > 0)] = roi1_dat[np.where(roi1_dat > 0)]
        opt_dat[np.where(roi2_dat > 0)] = np.max(roi1_dat) + 1

    opt_img = nib.Nifti1Image(opt_dat, roi1_aff, header = roi1_hdr)

    if opt_fln is None: 
        opt_fln = roi1_fln + '_merge.nii.gz'
    else: 
        opt_fln = opt_fln + '.nii.gz'

    nib.save(opt_img, opt_fln)
    
    return 1     


def roi_merge_n(roi_fln_list, opt_fln):

    try: 
        roi1_fln = roi_fln_list[0]
        roi2_fln = roi_fln_list[1]
        roi_merge(roi1_fln, roi2_fln, opt_fln)
        for roi_fln in roi_fln_list[2:]:
            roi_merge(opt_fln, roi_fln, opt_fln)
    except FileNotFoundError:
        print('ERROR: Cannot load images. ')



#
# LevelSet algorithm for ROI growing
#
def roi_levelset(ref_fln, sed_fln, opt_fln, num_ite = 20):

    sys.path.append('/Users/Sidong/Dropbox/Code/morphsnakes_py/')
    import morphsnakes

    try:      
        ref_img = nib.load(ref_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the REF image in NII.GZ. Trying to load the .NII instead.')
        ref_img = nib.load(ref_fln + '.nii')
    ref_dat = ref_img.get_data()
    ref_aff = ref_img.affine
    ref_hdr = ref_img.header
    
    try:
        sed_img = nib.load(sed_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        sed_img = nib.load(sed_fln + '.nii')
    sed_dat = sed_img.get_data()
    sed_aff = sed_img.affine
    sed_hdr = sed_img.header


    macwe = morphsnakes.MorphACWE(ref_dat, smoothing=1, lambda1=1, lambda2=2)
    # macwe.levelset = roi_dat
    macwe.levelset = circular_levelset(sed_dat.shape, (30, 50, 80), 25)

    for i in range(num_ite):
        macwe.step()

    opt_dat = macwe.levelset

    opt_img = nib.Nifti1Image(opt_dat, sed_aff, header = sed_hdr)
    nib.save(opt_img, opt_fln + '.nii.gz')

    return 1


#
# Graphcut algorithm for ROI growing
#
def roi_graphcut(ref_fln, sed_fln, opt_fln):

    sys.path.append('/Users/Sidong/Dropbox/Code/pysegbase/')
    import pycut as pspc

    try:      
        ref_img = nib.load(ref_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the REF image in NII.GZ. Trying to load the .NII instead.')
        ref_img = nib.load(ref_fln + 'nii')
    ref_dat = ref_img.get_data().astype(np.int16) * 20
    ref_aff = ref_img.affine
    ref_hdr = ref_img.header

    
    try:
        sed_img = nib.load(sed_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        sed_img = nib.load(sed_fln + '.nii')
    sed_dat = sed_img.get_data().astype(np.int16)
    sed_aff = sed_img.affine
    sed_hdr = sed_img.header

    sed_dat[sed_dat < 0] = 2
    sed_dat[sed_dat > 3.0] = 1


    igc = pspc.ImageGraphCut(ref_dat, voxelsize=[1, 1, 1])
    igc.set_seeds(sed_dat)
    igc.run()

    opt_dat = igc.segmentation

    opt_img = nib.Nifti1Image(opt_dat, sed_aff, header = sed_hdr)
    nib.save(opt_img, opt_fln + '.nii.gz')

    return 1



#
# GrowCut algorithm for ROI growing
#
def roi_growcut(ref_fln, sed_fln, opt_fln, adj_val):

    try:      
        ref_img = nib.load(ref_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the REF image in NII.GZ. Trying to load the .NII instead.')
        ref_img = nib.load(ref_fln + '.nii')
    ref_dat = ref_img.get_data()
    ref_aff = ref_img.affine
    ref_hdr = ref_img.header
    
    try:
        sed_img = nib.load(sed_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        sed_img = nib.load(sed_fln + '.nii')
    sed_dat = sed_img.get_data()
    sed_aff = sed_img.affine
    sed_hdr = sed_img.header

    opt_dat = sed_dat    

    sed_coords = np.where(sed_dat > 0)
    sed_values = ref_dat[sed_coords]

    # # the location of the seeding voxel
    sx = sed_coords[0][sed_values.argmax()]
    sy = sed_coords[1][sed_values.argmax()]
    sz = sed_coords[2][sed_values.argmax()]

    if adj_val is None:
        adj_val = (sed_values.max() - sed_values.min()) / 10.

    roi_min = sed_values.min() - adj_val
    roi_max = sed_values.max() + adj_val

    # Dimension of the input volume 
    dx, dy, dz = ref_dat.shape

    itr = 0
    # the local searching radius
    radius = 1
    
    while True:
        
        itr = itr + 1 
        # First stop criterion: reach the boundary of the image
        searching_extend    = np.array([itr + radius - sx, sx + itr + radius + 1 - dx, \
                                        itr + radius - sy, sy + itr + radius + 1 - dy, \
                                        itr + radius - sz, sz + itr + radius + 1 - dz])
        if (searching_extend >= 0).any(): 
            break

        # Second stop criterion: there is no new voxel with in the global value range 
        new_voxel_coords    = neighbor_coords(sx, sy, sz, itr)
        new_voxel_values    = ref_dat[new_voxel_coords[:, 0], new_voxel_coords[:, 1], new_voxel_coords[:, 2]]
        glb_voxel_indices   = np.where(np.logical_and(new_voxel_values < roi_max, new_voxel_values > roi_min))        
        
        if not glb_voxel_indices: 
            break
        
        else:
            for i in glb_voxel_indices[0]:
                lx, ly, lz          = new_voxel_coords[i, :] 
                patch_boolen        = opt_dat[lx - 1 : lx + 2, ly - 1 : ly + 2, lz - 1 : lz + 2]   

                if patch_boolen.sum() > 1:
                    local_value     = ref_dat[lx, ly, lz] 
                    patch_values    = ref_dat[lx - 1 : lx + 2, ly - 1 : ly + 2, lz - 1 : lz + 2] 
                    if patch_boolen.shape != patch_values.shape:
                      break
                    boolen_values   = patch_values[:] * patch_boolen[:]
                    
                    existing_values = boolen_values[np.where(boolen_values > 0)]
                    local_min       = existing_values.min() - adj_val
                    local_max       = existing_values.max() + adj_val                 
                    # Third stop criterion: the voxel value is beyond the range of local existing neighbors
                    if local_value < local_max and local_value > local_min:
                        opt_dat[lx, ly, lz] = 1

    pass

    opt_img = nib.Nifti1Image(opt_dat, sed_aff, header = sed_hdr)
    nib.save(opt_img, opt_fln + '.nii.gz')

    return 1

# #
# # Clustering the voxels in the file
# #
# def roi_cluster(roi_fln):

#     try:      
#         roi_img = nib.load(roi_fln + '.nii')
#     except FileNotFoundError:
#         print('INFO: Cannot find the .nii file, loading .nii.gz instead'
#         roi_img = nib.load(roi_fln + '.nii.gz')
#     roi_dat = roi_img.get_data()
#     roi_aff = roi_img.affine
#     roi_hdr = roi_img.header

#     opt_dat = np.zeros_like(roi_dat)
#     # Findout non zero voxels
#     pos_dat = np.nonzero(roi_dat)
#     # Construct the coordiante vectors
#     XX = pos_dat[0]
#     YY = pos_dat[1]
#     ZZ = pos_dat[2]
#     # convert the tuples to arrays, and transpose, so AffinityPropagation can read it
#     X = distance.squareform(distance.pdist(np.asarray(pos_dat).transpose(), 'cityblock'))
#     # Construct the mcluster
#     ap_model = cluster.AffinityPropagation(damping=.9, preference=-200)
#     ap_model.fit(X)
#     ap_centers = X[ap_model.cluster_centers_indices_]
#     ap_labels = model.labels_.astype(np.int)

#     for k in range(ap_centers.shape[0]):
#         roi_members = ap.labels == k
#         opt_dat[[XX[roi_members], YY[roi_members], ZZ[roi_members]]] = k + 1

#     opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
#     nib.save(opt_img, roi_fln + '_cluster.nii.gz')

#     return 1



#
# Normalize the ROI images
#
def roi_norm(roi_fln):

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    opt_dat = np.zeros_like(roi_dat)
    opt_dat = roi_dat - roi_dat.mean()
    opt_dat = opt_dat / roi_dat.std()
    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    nib.save(opt_img, roi_fln + '_norm.nii.gz')

    return 1

#
# Revese normalize the ROI images by choosing a threshold
#
def roi_norm_inv(roi_fln):

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    opt_dat = np.zeros_like(roi_dat)
    opt_dat = roi_dat.mean() - roi_dat
    opt_dat = opt_dat / roi_dat.std()
    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    nib.save(opt_img, roi_fln + '_norm_inv.nii.gz')

    return 1

#
# Add the values of two images given a ROI
#
def roi_add(ipt1_fln, ipt2_fln, roi_fln, opt_fln):

    try:      
        ipt1_img = nib.load(ipt1_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        ipt1_img = nib.load(ipt1_fln + '.nii')
    ipt1_dat = ipt1_img.get_data()
    ipt1_aff = ipt1_img.affine
    ipt1_hdr = ipt1_img.header

    try:
        ipt2_img = nib.load(ipt2_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        ipt2_img = nib.load(ipt2_fln + '.nii')
    ipt2_dat = ipt2_img.get_data()
    ipt2_aff = ipt2_img.affine
    ipt2_hdr = ipt2_img.header

    try:
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data().astype(int)
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header


    if not (ipt1_img.shape == ipt2_img.shape):
        print('ERROR: image dimension not match!!!')
        print('INFO: 1st input image ', ipt1_img.shape, ipt1_img.get_data_dtype())
        print('INFO: 2nd input image ', ipt2_img.shape, ipt2_img.get_data_dtype())
        return -1

    elif not (ipt1_img.shape == roi_img.shape):
        print('ERROR: image dimension not match mask dimesion!!!')
        print('INFO: input image ', ipt1_img.shape, ipt1_img.get_data_dtype())
        print('INFO: lesion mask ', roi_img.shape, roi_img.get_data_dtype())
        return -1

    else:
        opt_dat = np.zeros_like(ipt1_dat)
        for i in np.arange(np.max(roi_dat)) + 1:            
            roi_coords = np.where(roi_dat == i)
            opt_dat[roi_coords] = ipt1_dat[roi_coords] + ipt2_dat[roi_coords]
        
        opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
        nib.save(opt_img, opt_fln + '.nii.gz')
        return 1



#
# Substract the values of two images given a ROI
#
def roi_sub(ipt1_fln, ipt2_fln, roi_fln, opt_fln):

    try:      
        ipt1_img = nib.load(ipt1_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        ipt1_img = nib.load(ipt1_fln + '.nii')
    ipt1_dat = ipt1_img.get_data()
    ipt1_aff = ipt1_img.affine
    ipt1_hdr = ipt1_img.header

    try:
        ipt2_img = nib.load(ipt2_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        ipt2_img = nib.load(ipt2_fln + '.nii')
    ipt2_dat = ipt2_img.get_data()
    ipt2_aff = ipt2_img.affine
    ipt2_hdr = ipt2_img.header

    try:
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data().astype(int)
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header


    if not (ipt1_img.shape == ipt2_img.shape):
        print('ERROR: image dimension not match!!!')
        print('INFO: 1st input image ', ipt1_img.shape, ipt1_img.get_data_dtype())
        print('INFO: 2nd input image ', ipt2_img.shape, ipt2_img.get_data_dtype())
        return -1

    elif not (ipt1_img.shape == roi_img.shape):
        print('ERROR: image dimension not match mask dimesion!!!')
        print('INFO: input image ', ipt1_img.shape, ipt1_img.get_data_dtype())
        print('INFO: lesion mask ', roi_img.shape, roi_img.get_data_dtype())
        return -1

    else:
        opt_dat = np.zeros_like(ipt1_dat)
        for i in np.arange(np.max(roi_dat)) + 1:            
            roi_coords = np.where(roi_dat == i)
            opt_dat[roi_coords] = ipt1_dat[roi_coords] - ipt2_dat[roi_coords]
        
        opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
        nib.save(opt_img, opt_fln + '.nii.gz')
        return 1


#
# Multiply the values of two images given a ROI
#
def roi_multiply(ipt1_fln, ipt2_fln, roi_fln, opt_fln):

    try:      
        ipt1_img = nib.load(ipt1_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        ipt1_img = nib.load(ipt1_fln + '.nii')
    ipt1_dat = ipt1_img.get_data()
    ipt1_aff = ipt1_img.affine
    ipt1_hdr = ipt1_img.header

    try:
        ipt2_img = nib.load(ipt2_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        ipt2_img = nib.load(ipt2_fln + '.nii')
    ipt2_dat = ipt2_img.get_data()
    ipt2_aff = ipt2_img.affine
    ipt2_hdr = ipt2_img.header

    try:
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data().astype(int)
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header


    if not (ipt1_img.shape == ipt2_img.shape):
        print('ERROR: image dimension not match!!!')
        print('INFO: 1st input image ', ipt1_img.shape, ipt1_img.get_data_dtype())
        print('INFO: 2nd input image ', ipt2_img.shape, ipt2_img.get_data_dtype())
        return -1

    elif not (ipt1_img.shape == roi_img.shape):
        print('ERROR: image dimension not match mask dimesion!!!')
        print('INFO: input image ', ipt1_img.shape, ipt1_img.get_data_dtype())
        print('INFO: lesion mask ', roi_img.shape, roi_img.get_data_dtype())
        return -1

    else:
        opt_dat = np.zeros_like(ipt1_dat)
        for i in np.arange(np.max(roi_dat)) + 1:            
            roi_coords = np.where(roi_dat == i)
            opt_dat[roi_coords] = ipt1_dat[roi_coords] * ipt2_dat[roi_coords]
        
        opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
        nib.save(opt_img, opt_fln + '.nii.gz')
        return 1


#
# Calculate the overlap of two ROI images
#
def roi_overlap(roi1_fln, roi2_fln):

    try:      
        roi1_img = nib.load(roi1_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi1_img = nib.load(roi1_fln + '.nii')
    roi1_dat = roi1_img.get_data()
    roi1_aff = roi1_img.affine
    roi1_hdr = roi1_img.header


    try:
        roi2_img = nib.load(roi2_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi2_img = nib.load(roi2_fln + '.nii')
    roi2_dat = roi2_img.get_data()
    roi2_aff = roi2_img.affine
    roi2_hdr = roi2_img.header

    if not (roi1_img.shape == roi2_img.shape):
        print('ERROR: image dimension not match!!!')
        print('INFO: 1st lesion mask ', roi1_img.shape, roi1_img.get_data_dtype())
        print('INFO: 2nd lesion mask ', roi2_img.shape, roi2_img.get_data_dtype())
        return -1

    else:
        # The index of positive values in ROI 1, boolen
        roi1_pos = roi1_dat > 0
        # The index of positive values in ROI 2, boolen
        roi2_pos = roi2_dat > 0
        # Number of positive values in ROI 1
        nov_roi1 = roi1_pos.astype(int).sum()
        # Number of positive values in ROI 2
        nov_roi2 = roi2_pos.astype(int).sum()
        # The index of overlap voxels
        rois_and = np.logical_and(roi1_pos, roi2_pos)
        # The index of positive voxels
        rois_orr = np.logical_or(roi1_pos, roi2_pos)
        # Number of overlapped voxels
        nov_and = rois_and.astype(int).sum()
        # Number of merged voxels
        nov_orr = rois_orr.astype(int).sum()
        print(nov_and)
        return [nov_roi1, nov_roi2, nov_and, nov_orr]



#
# Calculate the correlation coefficient of two ROI images
#
def roi_corrcoef(ipt1_fln, roi1_fln, ipt2_fln, roi2_fln):

    try:      
        ipt1_img = nib.load(ipt1_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        ipt1_img = nib.load(ipt1_fln + '.nii')
    ipt1_dat = ipt1_img.get_data()
    ipt1_aff = ipt1_img.affine
    ipt1_hdr = ipt1_img.header

    try:      
        ipt2_img = nib.load(ipt2_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        ipt2_img = nib.load(ipt2_fln + '.nii')
    ipt2_dat = ipt2_img.get_data()
    ipt2_aff = ipt2_img.affine
    ipt2_hdr = ipt2_img.header    

    try:
        roi1_img = nib.load(roi1_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi1_img = nib.load(roi1_fln + '.nii')
    roi1_dat = roi1_img.get_data()
    roi1_aff = roi1_img.affine
    roi1_hdr = roi1_img.header

    try:
        roi2_img = nib.load(roi2_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi2_img = nib.load(roi2_fln + '.nii')
    roi2_dat = roi2_img.get_data()
    roi2_aff = roi2_img.affine
    roi2_hdr = roi2_img.header


    if not (roi1_img.shape == roi2_img.shape):
        print('ERROR: the dimension of the masks do not match!!!')
        print('INFO: 1st lesion mask ', roi1_img.shape, roi1_img.get_data_dtype())
        print('INFO: 2nd lesion mask ', roi2_img.shape, roi2_img.get_data_dtype())
        return -1

    if not (ipt1_img.shape == ipt2_img.shape):
        print('ERROR: the dimension of the images do not match!!!')
        print('INFO: 1st scan ', ipt1_img.shape, ipt1_img.get_data_dtype())
        print('INFO: 2nd scan ', ipt2_img.shape, ipt2_img.get_data_dtype())
        return -1

    if not (ipt1_img.shape == roi1_img.shape):
        print('ERROR: the dimension of the scan and mask do not match!!!')
        print('INFO: 1st scan ', ipt1_img.shape, ipt1_img.get_data_dtype())
        print('INFO: 2nd scan ', roi1_img.shape, roi1_img.get_data_dtype())
        return -1

    ipt1_coords = np.where(roi1_dat > 0)
    ipt1_values = ipt1_dat[ipt1_coords]

    ipt2_coords = np.where(roi2_dat > 0)
    ipt2_values = ipt2_dat[ipt2_coords]

    corrcoef = np.corrcoef(ipt1_values, ipt2_values)

    return corrcoef, ipt1_values, ipt2_values


#
# Calculate the mutual inforamtion of two ROI images
#
def roi_mutualinfo(ipt1_fln, roi1_fln, ipt2_fln, roi2_fln, bins):

    corrcoef, ipt1_values, ipt2_values = roi_corrcoef(ipt1_fln, roi1_fln, ipt2_fln, roi2_fln)
    mi = mutual_info(ipt2_values, ipt2_values, bins)
    return mi

#
# Calculate the dice coefficient between two ROIs
#
def roi_dice(roi1_fln, roi2_fln):

    try:      
        roi1_img = nib.load(roi1_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi1_img = nib.load(roi1_fln + '.nii')
    roi1_dat = roi1_img.get_data().astype(np.bool)


    try:
        roi2_img = nib.load(roi2_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi2_img = nib.load(roi2_fln + '.nii')
    roi2_dat = roi2_img.get_data().astype(np.bool)

    if not (roi1_img.shape == roi2_img.shape):
        print('ERROR: image dimension not match!!!')
        print('INFO: 1st lesion mask ', roi1_img.shape, roi1_img.get_data_dtype())
        print('INFO: 2nd lesion mask ', roi2_img.shape, roi2_img.get_data_dtype())
        return -1

    else:
        t = roi1_dat.sum() + roi2_dat.sum()
        if t == 0:
            dice = 0
        else:
            m = np.logical_and(roi1_dat, roi2_dat)
            dice = 2.* m.sum() / t
            # print [roi1_dat.sum(), roi2_dat.sum(), m.sum(), t]
    print(dice)
    return dice


###############################################
########### Supporting Functions ##############
###############################################


#
#  Cartesian 
#
def cartesian(arrays, out = None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype    
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype = dtype)    
    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)    
    if arrays[1:]:
        cartesian(arrays[1:], out = out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]    
    return out


#
# Search neighbor 
#
def search_neighbor(roi_nod_set, sed_nod, opt_nod_set, LAYER):    
    
    LAYER = LAYER + 1    
    sub_nod_set = roi_nod_set     
    
    if DEBUG:
        print('INFO: Current centre nod ', sed_nod, ' at Layer: ', LAYER)
    
    #     if the opt_nod_set includes the sed_nod
    if not sed_nod in opt_nod_set:
        #     take out the sed_nod from the roi_nod_set, and put it into the opt_nod_set  
        opt_nod_set.add(sed_nod)
        sub_nod_set.remove(sed_nod)
        #     initialize the neighbor nod set
        nbr_nod_set = set()
        #     All possible combinations of the displacement in the searching area
        disps = cartesian((np.arange(sed_nod[0]-1, sed_nod[0]+2), \
                           np.arange(sed_nod[1]-1, sed_nod[1]+2), \
                           np.arange(sed_nod[2]-1, sed_nod[2]+2)))
        disp_length = disps.shape[0]     
        for d in np.arange(disp_length):
            # the current displacement 
            disp_x0, disp_x1, disp_x2 = disps[d, :]
            nbr_nod_set.add((disp_x0, disp_x1, disp_x2))
        #     the matched nod set
        matched_nod_set = sub_nod_set.intersection(nbr_nod_set)
        #     if not able to find matched nods
        if not matched_nod_set:
            if DEBUG: print('STOP: Cannot find matched nod.')
        #     else keep searching 
        else:
            if DEBUG: print('INFO: Matched nod set', matched_nod_set)
            for matched_nod in matched_nod_set:
                search_neighbor(sub_nod_set, matched_nod, opt_nod_set, LAYER)
    return opt_nod_set, LAYER


#
# Search neighbor 
#

def neighbor_coords(sx, sy, sz, itr, out = None):

    vol_coords_yz = cartesian((np.array([sx-itr, sx+itr]), \
                                      np.arange(sy-itr, sy+itr+1), \
                                      np.arange(sz-itr, sz+itr+1)))

    vol_coords_xz = cartesian((np.arange(sx-itr+1, sx+itr), \
                                      np.array([sy-itr, sy+itr]), \
                                      np.arange(sz-itr, sz+itr+1)))

    vol_coords_xy = cartesian((np.arange(sx-itr+1, sx+itr), \
                                      np.arange(sy-itr+1, sy+itr), \
                                      np.array([sz-itr, sz+itr])))

    vol_coords = np.concatenate((vol_coords_yz, np.concatenate((vol_coords_xz, vol_coords_xy))))
    
    return vol_coords


#
# Create a circle seeding point for the level-set method
#
def circular_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u




#
# Calculate the mutual information
#
def mutual_info(x_values, y_values, bins):
    # Read in the data
    if not x_values.size == y_values.size:
        return 0
    else:
      c_xy = np.histogram2d(x_values, y_values, bins)[0]
      c_x = np.histogram(x_values, bins)[0]
      c_y = np.histogram(y_values, bins)[0]

      h_xy = self.shan_entropy(c_xy)
      h_x = self.shan_entropy(c_x)
      h_y = self.shan_entropy(c_y)

      mutualinfo = h_x + h_y - h_xy
      return mutualinfo


#
# Calculate the entropy of values
#
def shan_entropy(values):
    v_normalized = values / float(np.sum(values))
    v_normalized = v_normalized[np.nonzero(v_normalized)]
    shanentropy = -sum(v_normalized * np.log2(v_normalized)) 
    return shanentropy

###############################################
#####   Leg Muscle Supporting Functions   #####
#####              By Zihao               #####
###############################################

#
# Find the cut slice of the image
#
def crop_middle(roi_fln):
    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')

    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    projection_z = np.mean(roi_dat,axis=2)
    projection_z = np.rot90(projection_z)

    projection_zx = np.mean(projection_z, axis=0)
    left_max = np.argmax(projection_zx[0:int(np.shape(projection_zx)[0]/2)])
    right_max = np.argmax(projection_zx[int(np.shape(projection_zx)[0]/2):int(np.shape(projection_zx)[0])]) + int(np.shape(projection_zx)[0]/2)
    middle_point = np.argmin(projection_zx[left_max:right_max])
    middle_point += left_max

    # # plotting the result
    # fig = plt.figure(figsize=(6.5,3))

    # plt1 = fig.add_subplot(121)
    # plt2 = fig.add_subplot(122)
    # plt1.imshow(projection_z)
    # ind = 1
    # for i in projection_zx:
    #     plt2.scatter([ind],[i],c='red')
    #     ind += 1 

    # plt2.scatter(right_max, projection_zx[right_max], c='blue')
    # ind += 1
    # plt2.scatter(left_max, projection_zx[left_max], c='blue')
    # ind += 1
    # plt2.scatter(middle_point, projection_zx[middle_point], c='black')

    # plt1.plot([middle_point,middle_point], [0,roi_dat.shape[1]-1], 'k-')
    # plt1.axis('off')
    # plt2.plot([middle_point,middle_point], [0,8000], 'k-')
    # plt.show()


    print("******** cropping",roi_fln, "left ********")
    crop_command = ["fslroi", roi_fln, roi_fln+"_left.nii.gz", str(middle_point), str(roi_dat.shape[2]), '0', str(roi_dat.shape[2]), '85', '355']
    subprocess.call(crop_command)
    print("******** cropping",roi_fln, "right ********")
    crop_command = ["fslroi", roi_fln, roi_fln+"_right.nii.gz", '0', str(middle_point), '0', str(roi_dat.shape[2]), '85', '335']
    subprocess.call(crop_command)





