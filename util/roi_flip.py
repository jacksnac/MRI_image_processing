import sys
import roi_morph as mo

# roi_flip(roi_fln, ref_fln, opt_fln)
mo.roi_flip(sys.argv[1], sys.argv[2])

