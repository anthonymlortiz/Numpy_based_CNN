# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 18:09:03 2017

@author: Anthony
"""

import cnn_numpy as layer

def cnn_relu_forward(x, w, b, cnn_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.
  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = layer.cnn_backward_pass(x, w, b, cnn_param[0], cnn_param[1])
  out, relu_cache = layer.relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def cnn_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  cnn_cache, relu_cache = cache
  da = layer.relu_backward(dout, relu_cache)
  dx, dw, db = layer.cnn_backward_pass(da, cnn_cache)
  return dx, dw, db


def cnn_relu_pool_forward(x, w, b, stride, padding, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.
  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer
  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, cnn_cache = layer.cnn_forward_pass(x, w, b, stride, padding)
  s, relu_cache = layer.relu_forward(a)
  out, pool_cache = layer.max_pooling_forward_pass(s, pool_param)
  cache = (cnn_cache, relu_cache, pool_cache)
  return out, cache


def cnn_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  cnn_cache, relu_cache, pool_cache = cache
  ds = layer.max_pooling_backward_pass(dout, pool_cache)
  da = layer.relu_backward(ds, relu_cache)
  dx, dw, db = layer.cnn_backward_pass(da, cnn_cache)
  return dx, dw, db


def fully_connected_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU
  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = layer.fully_connected_forward(x, w, b)
  out, relu_cache = layer.relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def fully_connected_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = layer.relu_backward(dout, relu_cache)
  dx, dw, db = layer.fully_connected_backward(da, fc_cache)
  return dx, dw, db