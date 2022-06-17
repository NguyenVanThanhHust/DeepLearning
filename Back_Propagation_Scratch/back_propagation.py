import os
import numpy as np

def conv_forward(X, W):
    '''
    The forward computation of convolution function

    Arguments:
        X input matrix, numpy array, shape (H, W)
        W weight matrix, numpy array, shape(f, f) 
        
    Returns:
        out, output matrix, numpy array, shape (H, W)
        cache, cache of values needed to backward computation 
    '''
    # Retrieve dimensions from X's shape
    H_prev, W_prev = X.shape

    # REtriveing dimensions from W's shape
    (f, f) = W.shape

    # Compute output dimensions 
    H = H_prev - f + 1
    W = W_prev - f + 1

    # Initialize output matrix
    out = np.zeros((H, W))

    for h in range(H):
        for w in range(W):
            x_slice = X[h:h+f, w: w+f]
            out[h, w] = np.sum(x_slice * W)

    cache = (X, W)
    return out, cache


def conv_backward(dH, cache):
    '''
    The backward computation of convolution function
    Arguments:
        dH --gradient of the cost with respect to the conv layer
        cache --cache of values needed for backward
    Returns;
        dX -- gradient of the cost with respect to input  
        dW -- gradient of the cost with respect ot weight
    '''
    # Retrieve the information 
    (X, W) = cache

    # Get dimension from X shape
    H_prev, W_prev = X.shape

    # Get dimenson from W shape
    f, f = W.shape

    H, W = dH.shape

    dX = np.zeros(X.shape)
    dW = np.zeros(W.shape)

    for h in H:
        for w in W:
            dX[h:h+f, w:w+f] += W * dH(h, w)
            dW += X[h:h+f, w:w+f] * dH(h, w)

    return dX, dW
        