import sys
import roi_morph as mo

# crop an input image using a roi
mo.roi_crop(sys.argv[1], sys.argv[2])

