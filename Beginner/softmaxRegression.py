# -*- coding:utf-8 -*-

import tensorflow as tf

import input_data

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

p = tf.nn.softmax(tf.matmul(x, W) + b)
cost = -tf.reduce_sum(y*tf.log(p))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(100) :
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x : batch_xs, y : batch_ys})

pred = tf.equal(tf.argmax(p, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(pred, "float"))
print sess.run(accuracy, feed_dict={x : mnist.test.images, y : mnist.test.labels})
