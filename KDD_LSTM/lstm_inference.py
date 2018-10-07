# -*- coding: utf-8 -*-
import tensorflow as tf

rnn_unit = 20       #hidden layer units
layer_num = 2
input_size = 41
output_size = 2

keep_prob = tf.placeholder(tf.float32, []) 

weights={  
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit]),dtype=tf.float32),  
         'out':tf.Variable(tf.random_normal([rnn_unit,output_size]),dtype=tf.float32)  
        }  
biases={  
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit]),dtype=tf.float32),  
        'out':tf.Variable(tf.constant(0.1,shape=[output_size]),dtype=tf.float32)  
       } 
        
def lstm(X):
    batch_size = tf.shape(X)[0]  
    time_step = tf.shape(X)[1]  
    # in
    w_in = weights['in']  
    b_in = biases['in']    
    input = tf.reshape(X,[-1,input_size]) 
    input_rnn = tf.matmul(input,w_in)+b_in  
    input_rnn = tf.reshape(input_rnn,[-1,time_step,rnn_unit])
    # lstm
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_unit,state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
    init_state = mlstm_cell.zero_state(batch_size,dtype=tf.float32)  
    output_rnn,final_states = tf.nn.dynamic_rnn(mlstm_cell,input_rnn,initial_state=init_state,time_major=False)  
    # out
    output = tf.reshape(output_rnn,[-1,rnn_unit])
    w_out = weights['out']  
    b_out = biases['out']  
    pred = tf.nn.softmax(tf.matmul(output,w_out)+b_out)
    return pred