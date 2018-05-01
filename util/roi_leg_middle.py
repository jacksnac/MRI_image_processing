import sys
import roi_morph as mo

# Binarize the ROI image
mo.crop_middle(sys.argv[1])