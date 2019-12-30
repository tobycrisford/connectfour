# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:09:32 2019

@author: Toby
"""

import tensorflow as tf
import numpy as np
from train import train
from humanvmodel import humanvmodel

learning_rate = 0.0001
explore_rate = 4.75
memsize = 1000000
batchsize = 32
directory = "./model.ckpt"
chckptrate = 2000
discount = 0.8
#epsilon = 10**(-20)

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[84,None])
XB = tf.placeholder(tf.float32, shape=[84,None])

W1 = tf.get_variable("W1", [42, 84],initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", [21, 42],initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", [21, 21],initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable("W4", [7,21],initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.get_variable("b1", [42,1],initializer=tf.zeros_initializer())
b2 = tf.get_variable("b2", [21,1],initializer=tf.zeros_initializer())
b3 = tf.get_variable("b3", [21,1],initializer=tf.zeros_initializer())
b4 = tf.get_variable("b4", [7,1],initializer=tf.zeros_initializer())

Z1 = tf.add(tf.matmul(W1,X),b1)
A1 = tf.nn.relu(Z1)
Z2 = tf.add(tf.matmul(W2,A1),b2)
A2 = tf.nn.relu(Z2)
Z3 = tf.add(tf.matmul(W3,A2),b3)
A3 = tf.nn.relu(Z3)
Z4 = tf.add(tf.matmul(W4,A3),b4)
Y = Z4

Z1B = tf.add(tf.matmul(W1,XB),b1)
A1B = tf.nn.relu(Z1B)
Z2B = tf.add(tf.matmul(W2,A1B),b2)
A2B = tf.nn.relu(Z2B)
Z3B = tf.add(tf.matmul(W3,A2B),b3)
A3B = tf.nn.relu(Z3B)
Z4B = tf.add(tf.matmul(W4,A3B),b4)
YB = Z4B

ends = tf.placeholder(tf.float32, shape=[1,None])
wons = tf.placeholder(tf.float32, shape=[1,None])
moves = tf.placeholder(tf.int32, shape=[1,None])

Ym = tf.diag_part(tf.gather(Y, moves[0]))
YBmax = tf.reduce_max(YB, axis=0)

cost = (1/batchsize)*tf.reduce_sum(ends * tf.square(wons - Ym) + (1-ends)*tf.square(Ym - discount*YBmax))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

saver = tf.train.Saver()

init = tf.global_variables_initializer()

sess = tf.Session()

#sess.run(init)
saver.restore(sess, directory)

#train(sess, X, Y, optimizer, cost, X, XB, ends, wons, moves, explore_rate, memsize, batchsize, saver, directory, chckptrate, YBmax)
humanvmodel(sess, X, Y, humanfirst=True)

sess.close()