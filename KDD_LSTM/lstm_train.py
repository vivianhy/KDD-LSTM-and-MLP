# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import pandas as pd  
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import lstm_inference

lr = 0.001
batch_size = 100
time_step = 10
train_step = 2001
train_end = 400000

MODEL_SAVE_PATH = "/home/hy/KDD_LSTM/model/"
MODEL_NAME = "model.ckpt"

'''
def get_train_data(data):
    print("begin get train data")
    batch_index = []  
          
    scaler_for_x = MinMaxScaler(feature_range=(0,1))      
    scaled_x_data = scaler_for_x.fit_transform(data[:,:-1])   
    normalized_train_data = scaled_x_data[:]
    label_train = data[:,-1]   
           
    train_x,train_y = [],[]     
    for i in range(len(normalized_train_data)-time_step):  
        if i % batch_size==0:  
            batch_index.append(i) 
            
        # train_x    
        x = normalized_train_data[i:i+time_step,:41]  
        train_x.append(x.tolist())
        
        # train_y
        y = label_train[i:i+time_step]  
        labely = []
        for j in y:
            label_list = [0 for num in range(2)]
            label_list[int(j)] = 1
            labely.append(label_list)
        train_y.append(labely)  
        
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y
'''

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
    # input
    X = tf.placeholder(tf.float32,shape=[None,time_step,lstm_inference.input_size],name='x-input')  
    Y = tf.placeholder(tf.float32,shape=[None,time_step,lstm_inference.output_size],name='y-input')  
         
    pred = lstm_inference.lstm(X)  
    Y_ = tf.reshape(Y,[-1,lstm_inference.output_size])
    pred_ = tf.reshape(pred,[-1,lstm_inference.output_size])
    
    # loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = Y_,logits = pred_)
    loss = tf.reduce_mean(cross_entropy)  
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    
    # accuracy
    with tf.name_scope('evaluate'):
        correct_prediction = tf.equal(tf.argmax(pred_, 1), tf.argmax(Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    
    # batch_index,train_x,train_y = get_train_data(data)
    # print("get train data success!")
    # saver = tf.train.Saver()
    with tf.Session() as sess:  
        sess.run(tf.global_variables_initializer())
        # tensorboard
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/test",sess.graph)
        
        for i in range(train_step):
            xs,ys = next_batch(feature_train,label_train,batch_size*time_step)
            xs = tf.reshape(xs,[-1,time_step,lstm_inference.input_size])
            ys = tf.reshape(ys,[-1,time_step,lstm_inference.output_size])
            _,loss_value = sess.run([train_op,loss],feed_dict={X:xs.eval(),Y:ys.eval(),lstm_inference.keep_prob:1.0})
            
            if(i%50 == 0):
                xs_test = tf.reshape(feature_test,[-1,time_step,lstm_inference.input_size])
                ys_test = tf.reshape(label_test,[-1,time_step,lstm_inference.output_size])
                
                result = sess.run(merged,feed_dict={X:xs_test.eval(),Y:ys_test.eval(),lstm_inference.keep_prob:1.0})
                writer.add_summary(result,i)
                
                accuracy_value = sess.run(accuracy,feed_dict={X:xs_test.eval(),Y:ys_test.eval(),lstm_inference.keep_prob:1.0})
                print("step: %d ,loss on training batch: %g ,accuracy on test data: %g" % (i,loss_value,accuracy_value))
                # saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=i)
        '''
        for i in range(iter_num):  
            for step in range(len(batch_index)-1):
                begin = batch_index[step]
                end = batch_index[step+1]
                sess.run(train_op,feed_dict={X:train_x[begin:end],Y:train_y[begin:end],lstm_inference.keep_prob:1.0})
                print("iter %d ,step %d " % (i,step))
            if(i%2 == 0):
                loss_value,accuracy_value = sess.run([loss,accuracy],feed_dict={X:train_x[begin:end],Y:train_y[begin:end],lstm_inference.keep_prob:1.0})
                print("iter %d,loss is %g,accuracy is %g" % (i,loss_value,accuracy_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=i)
        '''
        
def main(argv=None):
    '''
    with (open(file_path,'r')) as f:
        df = pd.read_csv(f)
        data = df.iloc[:,:].values
    print(type(data))
    '''
    file_path = "/home/hy/KDD_MLP/kddtrain.csv"
    feature_train,label_train,feature_test,label_test = load_data(file_path)
    print("load data success")
    train(feature_train,label_train,feature_test,label_test)
    
if __name__ == '__main__':
    tf.app.run()