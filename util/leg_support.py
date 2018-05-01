import subprocess, os, sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import subprocess

from scipy import ndimage

DEBUG = False
DTYPE = np.int


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