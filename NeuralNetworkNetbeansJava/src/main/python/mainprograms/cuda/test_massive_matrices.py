#! ~/anaconda/bin/python2.7

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='llvmlite')

from numbapro import cuda, vectorize, guvectorize, check_cuda
from numbapro import void, uint8 , uint32, uint64, int32, int64, float32, float64, f8
import numpy as np


