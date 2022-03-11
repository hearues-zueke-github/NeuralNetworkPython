#! /opt/anaconda2/bin/python2.7

import pycuda.driver as cuda
import pycuda.autoinit , pycuda.compiler
import numpy

a = numpy.random.randn(4,4).astype(numpy.float32)
a.gpu = cuda.mem.alloc(a.nbytes)
cuda.memcpy.htod(a.gpu, a)
