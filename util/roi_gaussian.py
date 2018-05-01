import sys
import roi_morph as mo

# crop an input image using a roi
mo.roi_gaussian(sys.argv[1], kernel_size = sys.argv[2])

