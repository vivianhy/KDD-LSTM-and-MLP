# -*- coding: utf-8 -*-
import tensorflow as tf

INPUT_NODE = 41
OUTPUT_NODE = 2
LAYER1_NODE = 30

'''
def get_weight_variable(shape):
	weights = tf.Variable(tf.random_normal([INPUT_NODE, OUTPUT_NODE]), name='weights')
	 if regularizer != None:
		 tf.add_to_collection('losses',regularizer(weights))
	return weights
'''

def inference(input_tensor):
	with tf.variable_scope('layer1'):
		weights = tf.Variable(tf.random_normal([INPUT_NODE, LAYER1_NODE]), name='weights')
		biases = tf.Variable(tf.zeros([LAYER1_NODE]), name='biases')
		layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
	
	with tf.variable_scope('layer2'):
		weights = tf.Variable(tf.random_normal([LAYER1_NODE, OUTPUT_NODE]), name='weights')
		biases = tf.Variable(tf.zeros([OUTPUT_NODE]), name='biases')
		layer2 = tf.matmul(layer1,weights)+biases

	return layer2
