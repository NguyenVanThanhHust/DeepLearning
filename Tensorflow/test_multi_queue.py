import tensorflow as tf

#simulate with some raw input data, 3 samples of 1 data point
x_input_data = tf.random_normal([3], mean = -1, stddev = 4)

# we build a FIFO queue inside a graph
q = tf.FIFOQueue(capacity = 3, dtypes = tf.float32)

#check data
x_input_data = tf.Print(x_input_data, data=[x_input_data], message="Raw inputs data generated:", summarize=6)
# fill the queue with our data
enqueue_op = q.enqueue_many(x_input_data)

# To leverage multi-threading we create a "QueueRunner"
# that will handle the "enqueue_op" outside of the main thread
# We don't need much parallelism here, so we will use only 1 thread
numberOfThreads = 1 
qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
# Don't forget to add your "QueueRunner" to the QUEUE_RUNNERS collection
tf.train.add_queue_runner(qr) 

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
    # But now we build our coordinator to coordinate our child threads with
    # the main thread
    coord = tf.train.Coordinator()

    # Beware, if you don't start all your queues before runnig anything
    # The main threads will wait for them to start and you will hang again
    # This helper start all queues in tf.GraphKeys.QUEUE_RUNNERS
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(y)
    sess.run(y) 
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)

    # We request our child threads to stop ...
    coord.request_stop()
    # ... and we wait for them to do so before releasing the main thread
    coord.join(threads)