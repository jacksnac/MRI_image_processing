import sys
import roi_morph as mo

# Binarize the ROI image
mo.roi_dilate_n(sys.argv[1], num_shl = sys.argv[2], option = sys.argv[3])
