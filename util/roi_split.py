import sys
import roi_morph as mo

# Split the multi-value lesion masks into individual single-value masks
mo.roi_split(sys.argv[1])





