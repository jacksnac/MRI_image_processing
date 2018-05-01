import sys
import roi_morph as mo

# Binarize the ROI image
mo.roi_closing(sys.argv[1], option = sys.argv[2])
