import sys
import roi_morph as mo

# Binarize the ROI image
mo.roi_bin(sys.argv[1])
