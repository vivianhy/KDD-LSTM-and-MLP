# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import mlp_inference
import mlp_train

'''
def load_data():
    featurex = []
    labely = []
    file_path = "/home/hy/KDD_MLP/kddtest.csv"
    with (open(file_path,'r')) as data_from:
        csv_reader = csv.reader(data_from)
        for j in csv_reader:
            temp = [float(n) for n in j]
            featurex.append(temp[:41])
            label_list = [0 for num in range(2)]
            label_list[int(j[41])] = 1
            labely.append(label_list)
    return featurex,labely
'''

def load_data(file_path):   
    with (open(file_path,'r')) as f:
        df = pd.read_csv(f)
        data = df.iloc[:,:].values
        
    scaler_for_x = MinMaxScaler(feature_range=(0,1))      
    scaled_x_data = scaler_for_x.fit_transform(data[:,:-1]) 
    featurex = scaled_x_data.tolist()    
    
    label_test = data[:,-1]
    labely = []
    for i in label_test:
        label_list = [0 for num in range(2)]
        label_list[int(i)] = 1
        labely.append(label_list)
        
    return featurex,labely
    
def evaluate(feature,label):
    with tf.Graph().as_default() as g:
        # input
        x = tf.placeholder(tf.float32,[None,mlp_inference.INPUT_NODE],name='x-input')
        y_ = tf.placeholder(tf.float32,[None,mlp_inference.OUTPUT_NODE],name='y-input')
        validate_feed = {x:feature,y_:label}
        
        # accuracy
        y = mlp_inference.inference(x)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        variable_averages = tf.train.ExponentialMovingAverage(mlp_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mlp_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                print("step: %s ,test accuracy: %g" % (global_step,accuracy_score))
            else:
                print('no checkpoint file found')
                return
                
def main(argv=None):
    file_path = "/home/hy/KDD_MLP/kddtest.csv"
    feature,label = load_data(file_path)
    print("load data success")
    evaluate(feature,label)
    
if __name__ == '__main__':
    tf.app.run()
        
