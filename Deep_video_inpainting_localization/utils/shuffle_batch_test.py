import tensorflow as tf
import tensorflow.contrib as contrib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# dataset = tf.data.Dataset.range(13).batch(4, drop_remainder=True).shuffle(buffer_size=100).repeat(2)
dataset = tf.data.Dataset.range(13).shuffle(buffer_size=100).batch(4, drop_remainder=True).repeat(2)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(6):
        value = sess.run(next_element)
        print(value)



# inputs = tf.placeholder(dtype=tf.float32, shape=[1, 4, 10, 10, 28])  # [batch_size,time_setp,width,high,channeals] 5D
# cell = contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[10, 10, 28], output_channels=6, kernel_shape=[3, 3])
# initial_state = cell.zero_state(batch_size=4, dtype=tf.float32)
# output, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, time_major=True, initial_state=initial_state)
# print(output)
# print(final_state)
