# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 12:25:07 2017
numpy based CNN, lets get the hands dirty

@author: Anthony Ortiz
"""
#X-> Input (X: W x H x B x I)
# C <- nb of columns or Width of input image
# R <- Height of input image
# B <- Number of bands or chanels (RBG = 3, HSI >= 3, ~=270 )
# N <- Number of images for training
# We assume all input images have same size

#W-> Weights (W: F x B x FH x FW )
# F <- number of filters
# B <- Number of bands or chanels (RBG = 3, HSI >= 3, ~=270 )
# FH <- Filter Height
# FW <- Filter Width

#b-> bias
from im2col import im2col_indices
from im2col import col2im_indices
import numpy as np
def cnn_forward_pass(X, W, b, stride = 1, padding =1):
  out = None
  N, B, R, C   = X.shape
  F, _, HH, WW = W.shape

  # Dimensionality check
  assert ( R + 2 * padding - HH) % stride == 0, 'width doesn\'t work with current parameter setting'
  assert ( C + 2 * padding - WW) % stride == 0, 'height doesn\'t work with current parameter setting'

  # Initialize output
  out_H = ( R + 2 * padding - HH) / stride + 1
  out_W = ( C + 2 * padding - WW) / stride + 1
  out = np.zeros( (N, F, out_H, out_W), dtype=X.dtype ) 

  x_cols = im2col_indices(X, HH, WW, padding, stride)

  res = W.reshape((W.shape[0], -1)).dot(x_cols) + b[:, np.newaxis]

  out = res.reshape((F, out_H, out_W, N))
  out = out.transpose(3, 0, 1, 2)

  cache = (X, W, b, stride, padding, x_cols)
  return out, cache
  
def cnn_backward_pass(dout, cache, debug=False):
  """
  A naive implementation of the backward pass for a convolutional layer.
  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, stride, padding) as in cnn_backward_pass
  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """

  dx, dw, db = None, None, None

  x, w, b, stride, padding, x_cols = cache

  db = np.sum( dout, axis=(0, 2, 3) )
  F, _, HH, WW = w.shape

  dout_reshape = np.reshape(dout.transpose(1,2,3,0), (F, -1))

  dw = dout_reshape.dot(x_cols.T).reshape(w.shape)

  dx_cols = w.reshape(F, -1).T.dot(dout_reshape)

  dx = col2im_indices(dx_cols, x.shape, field_height=HH, field_width=WW, padding=padding, stride=stride, verbose=False)

  if debug:
    print "dout's shape: {}".format( str(dout.shape) ) 
    print "dout's reshape: {}".format( str(dout_reshape.shape))
    print "x's shape: {}".format( str(x.shape) )
    print "x's cols: {}".format( str(x_cols.shape))
    print "w's shape: {}".format( str(w.shape) )
    print "b's shape: {}".format( str(b.shape) )
    print "stride: {}".format( str(stride) )
    print "padding: {}".format( str(padding) )


  return dx, dw, db
  
def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).
  Input:
  - x: Inputs, of any shape
  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  out = x.copy()
  # ReLU non-linearity
  out[out < 0] = 0

  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).
  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout
  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  dx = dout.copy()

  # Filter non-positive activation's gradient
  dx[x <= 0] = 0
  return dx

def max_pooling_forward_pass(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.
  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions
  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """

  N, B, R, C = x.shape

  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # First validate the pooling parameters
  assert R % pool_height == 0, "Image height not divisible by pooling height"
  assert C % pool_width == 0, "Image width not divisible by pooling width"

  out = np.zeros((N, B, R / pool_height, C / pool_width))

  # Pooling layer forward using iterative method
  for ii, i in enumerate(xrange(0, R, stride)):
    for jj, j in enumerate(xrange(0, C, stride)):
      # iterate through each central point
      out[:, :, ii, jj] = np.amax( x[:, :, i:i+pool_height,j:j+pool_width].reshape(N, B, -1), axis=2)

  cache = (x, pool_param)
  return out, cache
  
def max_pooling_backward_pass(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.
  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.
  Returns:
  - dx: Gradient with respect to x
  """

  # unpack layer cache
  x, pool_param = cache

  N, B, R, C = x.shape
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  dx = np.zeros_like(x)
  # Pooling layer backward using iterative method
  for ii, i in enumerate(xrange(0, R, stride)):
    for jj, j in enumerate(xrange(0, C, stride)):
      max_idx = np.argmax( x[:, :, i:i+pool_height,j:j+pool_width].reshape(N, B, -1), axis=2)

      max_cols = np.remainder(max_idx, pool_width) + j
      max_rows = max_idx / pool_width + i

      for n in xrange(N):
        for b in xrange(B):
          dx[n, b, max_rows[n, b], max_cols[n, b]] += dout[n, b, ii, jj]


  dx = dx.reshape(N, B, R, C)

  return dx
  

def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

def fully_connected_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.
  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i
  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None

  # reshape input into rows
  out = x.reshape( x.shape[0], np.prod(x.shape[1:]) )
  # Linear activation 
  out = out.dot(w) + b[np.newaxis, :]

  cache = (x, w, b)
  return out, cache


def fully_connected_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.
  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################

  sp = x.shape

  x  = np.reshape( x, ( sp[0] , np.prod(sp[1:]) ) )
  dw = np.dot( x.T, dout )
  db = np.sum( dout, axis=0 )
  dx = np.reshape( np.dot( dout, w.T ), sp )

  return dx, dw, db

