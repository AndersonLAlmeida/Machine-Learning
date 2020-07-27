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
# Result: Three rows and four columns, now it's a matrix, ot a vector.
# The shape got changed, but the size and numbes (inside) are the same
# array([[ 0.,  1.,  2.,  3.],
#        [ 4.,  5.,  6.,  7.],
#        [ 8.,  9., 10., 11.]])

x = x.reshape(3,-1)
x = x.reshape(-1,4)
# Result: The same as the last example, although we just need to put the
# quantity of rows or columns that we want, this case should work when
# we don't know how many columns or rows could have and we just know
# how many columns or rows we want

np.zeros((2, 3, 4))
# Result: Matrice of zeros with a shape of (2,3,4)
# array([[[0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.]],
#
#       [[0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.]]])

np.ones((2, 3, 4))
# Result: Matrice of zeros with a shape of (2,3,4)
# array([[[1., 1., 1., 1.],
#        [1., 1., 1., 1.],
#        [1., 1., 1., 1.]],
#
#       [[1., 1., 1., 1.],
#        [1., 1., 1., 1.],
#        [1., 1., 1., 1.]]])


np.random.normal(0, 1, size=(3, 4))
# "Often, we want to randomly sample the values for each element
# in a tensor from some probability distribution"
# Addendum: Each of its elements is randomly sampled from a standard Gaussian
# (normal) distribution with a mean of 0 and a standard deviation of 1.
# Result: Matrice of ramdomly numbers with a shape of (3,4)
# array([[ 2.2122064 ,  1.1630787 ,  0.7740038 ,  0.4838046 ],
#       [ 1.0434405 ,  0.29956347,  1.1839255 ,  0.15302546],
#       [ 1.8917114 , -1.1688148 , -1.2347414 ,  1.5580711 ]])


np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# Result: A simple array declared
# array([[2., 1., 4., 3.],
#       [1., 2., 3., 4.],
#       [4., 3., 2., 1.]])


#************************** OPERATIONS ************************#

# Elementwise operations: These apply a standard scalar operation to each
# element of an array
