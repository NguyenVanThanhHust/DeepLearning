from __future__ import  absolute_import
import os

import tensorflow as tf
import sonnet as snt
import numpy as np 
import matplotlib.pyplot as plt

# tf.enable_v2_behavior()

print("TensorFlow version {}".format(tf.__version__))
print("Sonnet version {}".format(snt.__version__))

def gaussian_mask(u, s, d, R, C);
    """
    u: tf.Tensor, center of first Gaussian
    s: tf.Tensor, standard deviation of Gaussian
    d: tf.Tensor, shiftbetween Gaussian center
    R: int, number of rows in the mask, there is one Gaussian per row
    C: int, number of cols in the mask
    """
    # indices to create center
    R = tf.to_float(tf.reshape(tf.range(R), (1,1,R))) 
    # create an array type int [0, 1,.., R-1] than reshape to (1, 1, R) and change to float
    C = tf.to_float(tf.reshape(tf.range(C), (1,C,1)))
    centres = u[np.newaxis, :, np.newaxis] + R * d
    column_centres = C-centres
    mask = tf.exp(-0.5 * tf.square(column_centres/s))
    # add eps for numerical stability
    normalised_maks = mask/(tf.reduce_sum(mask, 1, keep_dims=True) + 1e-8)
    return normalised_maks

def gaussian_glimpse(img_tensor, transform_params, crop_size):
    """
    :param img_tensor: tf.Tensor of size (batch_size, Height, Width, channels)
    :param transform_params: tf.Tensor of size (batch_size, 6), where params are  (mean_y, std_y, d_y, mean_x, std_x, d_x) specified in pixels.
    :param crop_size): tuple of 2 ints, size of the resulting crop
    """
    # parse arguments
    h, w = crop_size
    H, W = img_tensor.shape.as_list()[1:3]
    split_ax = transform_params.shape.ndims -1
    uy, sy, dy, ux, sx, dx = tf.split(transform_params, 6, split_ax)
    # create Gaussian masks, one for each axis
    Ay = gaussian_mask(uy, sy, dy, h, H)
    Ax = gaussian_mask(ux, sx, dx, w, W)
    # extract glimpse
    glimpse = tf.matmul(tf.matmul(Ay, img_tensor, adjoint_a=True), Ax)
    return glimpse

def spatial_transformer(img_tensor, transform_params, crop_size):
    """
    img_tensor: tf.Tensor of size(batch_size, Height, Width, channels)
    transform_params: tf.Tensor of size(batch_size, 4) where params are  (scale_y, shift_y, scale_x, shift_x)
    crop_size: tuple of 2 ints, size of resulting crop
    """
    constrains = snt.AffineWarpConstraints.no_shear_2d()
    img_size = img_tensor.shape.as_list()[1:]
    warper = snt.AffineGridWarper(img_size, crop_size, constrains)
    grid_coords = warper(transform_params)
    glimpse = snt.resampler(img_tensor[..., tf.newaxis], grid_coords)
    return glimpse

if __name__ == '__main__':
