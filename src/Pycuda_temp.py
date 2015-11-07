import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler as SourceModule
import numpy as np
import pycuda.gpuarray as gpuarray

a=gpuarray.to_gpu(np.ones((10000,10000)).astype(np.float32))
print a
b=gpuarray.to_gpu(np.ones((10000,10000)).astype(np.float32))
print b

mod = pycuda.compiler.SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }""")
r = (2*a + b).get()
#func = mod.get_function("doublify")
#func(cuda.InOut(a), block=(4,4,1))
print r


