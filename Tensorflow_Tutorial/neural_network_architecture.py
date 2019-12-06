import tensorflow as tf 

#generate random inputd
def generate_random_input(num_sam = 10000 , dim = 128):
    """
    This function is to generate random input with num_sam samples which each sample has dim dimension

    Args:
        num_sam : number of samples , default = 10000
        dim : dimension of each sample, defualt = 128

    Return :
        dataset contain data point and label

    Note : in this case, we predict simple rule as follow :
        - Divide input as 16 part, if sum of x in each part the
        - predict 1 if the sum of the elements is positive and 0 otherwise
    """
    X_inputs_data = tf.random_normal([num_sam, dim], mean = 0, stddev = 4)
    y_inputs_data = tf.cast(tf.reduce_sum(x_inputs_data, axis=1, keep_dims=True) > 0, tf.int32)
    # range_sum = []
    # for i in range(16):
        # sum_checkpoint = -4 + i *0.5
        # range_sum.append(sum_checkpoint)
    # range_sum.append(4)
    # for i in range(num_sam):
        # sum_x = tf.reduce_sum(x_inputs_data, axis=1, keep_dims=True)
        # for i in range(16):
            # if (sum_x >= range_sum[i] and sum_x < range_sum[i+1]):
                # Y_inputs_data.append(i)
                # break;
            # if sum_x == 
    return (X_inputs_data, Y_inputs_data)

class NetworkArchitecture():
