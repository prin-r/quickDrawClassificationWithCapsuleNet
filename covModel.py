import tensorflow as tf
import os
import numpy as np
import classificationTesting
import random

trainingSet1 = []
inputSize = 16
classNum = 6

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def getTestLabel2():
    return trainingSet1[0], trainingSet1[1]

def getTestLabel():
    batchSize = 7

    listRand =  [
                    [tf.constant([[0, 1, 1, 0],[1, 1, 0, 1],[1, 1, 1, 0],[1, 1, 0, 1]], dtype=tf.float32) , tf.constant([[1, 0, 0, 0]], dtype=tf.float32) ],
                    [tf.constant([[-1, 1, -1, 1], [1, -1, 1, -1], [0, -1, 1, 0], [0, 1, 1, -1]], dtype=tf.float32),tf.constant([[0, 1, 0, 0]], dtype=tf.float32)],
                    [tf.constant([[0, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 0, 1]], dtype=tf.float32),tf.constant([[0, 0, 1, 0]], dtype=tf.float32)],
                    [tf.constant([[1, 1, 0, 1], [0, -1, 1, 1], [1, 1, 1, 1], [1, -1, -1, 1]], dtype=tf.float32),tf.constant([[0, 0, 0, 1]], dtype=tf.float32)]
                ]

    e = random.choice(listRand)
    ii = tf.reshape(e[0], [1, 16])
    oo = tf.reshape(e[1], [1, 4])

    for j in range(batchSize):
        ee = random.choice(listRand)
        ii = tf.concat([ii, tf.reshape(ee[0], [1, 16])], 0)
        oo = tf.concat([oo, tf.reshape(ee[1], [1, 4])], 0)

    return ii , oo

def lrelu(x, alpha):
  return tf.maximum(x, alpha * x)

def makeModel():

    x_ = tf.placeholder(tf.float32, shape=[None, inputSize * inputSize])
    y_ = tf.placeholder(tf.float32, shape=[None, classNum])

    x_image = tf.reshape(x_, [-1, inputSize, inputSize, 1], name='image')

    #layer1
    W_conv1 = weight_variable([5, 5, 1, 4])
    b_conv1 = bias_variable([4])

    h_conv1 = lrelu(conv2d(x_image, W_conv1) + b_conv1, 0.2)

    # layer2
    W_conv2 = weight_variable([5, 5, 4, 8])
    b_conv2 = bias_variable([8])

    h_conv2 = lrelu(conv2d(h_conv1, W_conv2) + b_conv2, 0.2)

    # layer3
    W_conv3 = weight_variable([5, 5, 8, 16])
    b_conv3 = bias_variable([16])

    h_conv3 = lrelu(conv2d(h_conv2, W_conv3) + b_conv3, 0.2)

    h_flat = tf.reshape(h_conv3, [-1, 4 * 4 * 16])

    #layer4
    W_fc1 = weight_variable([4 * 4 * 16, 4])
    b_fc1 = bias_variable([4])

    h_fc1 = tf.nn.softmax(tf.matmul(h_flat, W_fc1) + b_fc1)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=h_fc1))

    #train_step = tf.train.AdamOptimizer(2e-4).minimize(cross_entropy)
    train_step = tf.train.MomentumOptimizer(0.01, 0.9).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(h_fc1, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(20000):
        bi , bo = getTestLabel2()
        if i % 25 == 0:
            train_accuracy = accuracy.eval(feed_dict={x_: bi, y_: bo})
            print('step %d, training accuracy %g' % (i, train_accuracy))

        if i % 100 == 0:
          print (bo)
          print (sess.run(h_fc1, {x_: bi, y_: bo}))

        train_step.run(feed_dict={x_: bi, y_: bo})








trainingSet1 = classificationTesting.makeTrainingSet()
"""
b1 = np.reshape(trainingSet1[0], [20,16])
b2 = np.reshape(trainingSet1[1], [20,2])

print (b1)
print (b2)
"""

makeModel()