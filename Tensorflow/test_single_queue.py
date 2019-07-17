import tensorflow as tf

#simulate with some raw input data, 3 samples of 1 data point
x_input_data = tf.random_normal([3], mean = -1, stddev = 4)

# we build a FIFO queue inside a graph
q = tf.FIFOQueue(capacity = 3, dtypes = tf.float32)

# fill the queue with our data
enqueue_op = q.enqueue_many(x_input_data)

# deque op to get the next elements in the queue following the FIFO policy
input = q.dequeue()

# the input tensor is the equivalent of a placeholder now
# but directly connected to teh data sources in the graph

# Each time we use the input tensor, we print tehe number of elements left 
input = tf.Print(input, data = [q.size()], message = "Number elements left : ")

#fake graph : START
y = input + 1
#fake graph : END

# start the session
with tf.Session() as sess:
    sess.run(enqueue_op)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)