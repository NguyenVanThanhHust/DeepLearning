import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# #define a variable
# x = tf.Variable(0, dtype = tf.int32)

# # We use a simple assign operation
# assign_op = tf.assign(x,x + 1)

# with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
  # sess.run(tf.global_variables_initializer())
  
  # for i in range(5):
    # print('x:', sess.run(x))
    # sess.run(assign_op)
	

# # define a "shape able" variable
# x = tf.Variable([], dtype = tf.int32, 
                    # validate_shape = False, # to change shape later
                    # trainable = False
                    # )

# #build new shape and assign items
# concat = tf.concat([x, [0]], 0)
# assign_op = tf.assign(x, concat, validate_shape = False)

# with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    # sess.run(tf.global_variables_initializer())
    # for i in range(5):
        # print('x:', sess.run(x), 'shape:', sess.run(tf.shape(x)))
        # sess.run(assign_op) 
		
		

# We define our Variables and placeholders
x = tf.placeholder(tf.int32, shape=[], name='x')
y = tf.Variable(2, dtype=tf.int32)

# We set our assign op
# assign_op = tf.assign(y, y + 1)

# We build our multiplication (this could be a more complicated graph)
out = x * y

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  
  for i in range(3):
    print('output:', sess.run(out, feed_dict={x: 1}))