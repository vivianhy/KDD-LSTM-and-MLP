# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import mlp_inference

BATCH_SIZE = 1000
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
# REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 2001
MOVING_AVERAGE_DECAY = 0.99
train_end = 400000

MODEL_SAVE_PATH = "/home/hy/KDD_MLP/model/"
MODEL_NAME = "model.ckpt"

def load_data(file_path):   
    with (open(file_path,'r')) as f:
        df = pd.read_csv(f)
        data = df.iloc[:,:].values
        
    # normalize data    
    scaler_for_x = MinMaxScaler(feature_range=(0,1))      
    scaled_x_data = scaler_for_x.fit_transform(data[:,:-1])

    # feature for train and test
    featurex_train = scaled_x_data[0:train_end].tolist()
    featurex_test = scaled_x_data[train_end:].tolist()
    
    # label for train and test
    label_train = data[0:train_end,-1]
    label_test = data[train_end:,-1]
    
    labely_train = []
    for i in label_train:
        label_list = [0 for num in range(2)]
        label_list[int(i)] = 1
        labely_train.append(label_list)
        
    labely_test = []
    for i in label_test:
        label_list = [0 for num in range(2)]
        label_list[int(i)] = 1
        labely_test.append(label_list)
        
    return featurex_train,labely_train,featurex_test,labely_test
    
def next_batch(feature_list,label_list,size):
    feature_batch_temp = []
    label_batch_temp = []
    f_list = random.sample(range(len(feature_list)), size)
    for i in f_list:
        feature_batch_temp.append(feature_list[i])
    for i in f_list:
        label_batch_temp.append(label_list[i])
    return feature_batch_temp,label_batch_temp

def train(feature_train,label_train,feature_test,label_test):
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32,[None,mlp_inference.INPUT_NODE],name='x-input')
        y_ = tf.placeholder(tf.float32,[None,mlp_inference.OUTPUT_NODE],name='y-input')
    
    # regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mlp_inference.inference(x)
    global_step = tf.Variable(0,trainable=False)
    
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss',loss)
        # loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        
    with tf.name_scope('train'):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,len(feature_train)/BATCH_SIZE,LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')
        
    with tf.name_scope('evaluate'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # tensorboard
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/test",sess.graph)
        
        for i in range(TRAINING_STEPS):
            xs,ys = next_batch(feature_train,label_train,BATCH_SIZE)
            _,loss_value,step= sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            
            '''
            if(i%200 == 0):
                accuracy_value = sess.run(accuracy,feed_dict={x:feature_test,y_:label_test})
                print("step: %d ,loss on training batch: %g ,accuracy on test data: %g" % (i,loss_value,accuracy_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=i)
            '''
            
            if(i%50 == 0):
                result = sess.run(merged,feed_dict={x:feature_test,y_:label_test})
                writer.add_summary(result,i)
                #print(i)

def main(argv=None):
    file_path = "/home/hy/KDD_MLP/kddtrain.csv"
    feature_train,label_train,feature_test,label_test = load_data(file_path)
    print("load data success")
    train(feature_train,label_train,feature_test,label_test)
    
if __name__ == '__main__':
    tf.app.run()
    
