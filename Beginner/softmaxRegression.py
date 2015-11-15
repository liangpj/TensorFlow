# -*- coding:utf-8 -*-

import tensorflow as tf

import input_data

## 导入数据集
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

##  plaoceholder(Dtype, shape)  创建一个2D的张量(tensor)
## Dtype： 数据类型
## shape: 维度,其中 None表示该维为可以为任意长度
##
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])


## Variable 是创建一个变量，在训练过程中该变量的值是需要改变的
## 注意与placeholder区别
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

## 预测模型和cost
p = tf.nn.softmax(tf.matmul(x, W) + b)
cost = -tf.reduce_sum(y*tf.log(p))

## 也许tensorflow的优势就在这里吧。
## 我们只需要bp过程，bp过程tensorflow自动进行求解和计算
## 0.01 为学习率，而cost是我们希望最小化的值
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

## 初始化所有的变量
init = tf.initialize_all_variables()

## 创建一个会话
## 因为所有的操作都是在会话里进行的
sess = tf.Session()
sess.run(init)

## 梯度训练
for i in range(100) :
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x : batch_xs, y : batch_ys})

## 模型评估
pred = tf.equal(tf.argmax(p, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(pred, "float"))
print sess.run(accuracy, feed_dict={x : mnist.test.images, y : mnist.test.labels})
