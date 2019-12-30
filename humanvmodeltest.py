# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 20:50:01 2019

@author: Toby
"""

from humanvmodel import humanvmodel
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[84,None])

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
Y = tf.sigmoid(Z4)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

humanvmodel(sess,X,Y)

sess.close()