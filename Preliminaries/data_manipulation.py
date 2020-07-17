# Frameworks:

# - MxNet
from mxnet import np, npx
npx.set_np()

# - Pytorch
import torch

# - Tensorflow
import tensorflow as tf

# I choosed mxnet because my study's group is using mxnet
# np (numpy) and npx (numpy_extension)

# A tensor represents a (possibly multi-dimensional) array of numerical values.
# With one axis, a tensor corresponds (in math) to a vector. With two axes,
# a tensor corresponds to a matrix. Tensors with more than two axes do not have
# special mathematical names.

#                              Code Examples                                  #

x = np.range(12)
x
# Result: array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])

x.shape
# Result: (12,) - twelve rows and one column

x.size
# Result: 12

x = x.reshape(3, 4)
# Result: three rows and four columns, now it's a matrix, ot a vector.
# The shape got changed, but the size and numbes (inside) are the same
# array([[ 0.,  1.,  2.,  3.],
#        [ 4.,  5.,  6.,  7.],
#        [ 8.,  9., 10., 11.]])



