import sys
import roi_morph as mo

# Binarize the ROI image
mo.roi_bin(sys.argv[1])
mo.roi_flip(sys.argv[1] + '_bin', sys.argv[2])
