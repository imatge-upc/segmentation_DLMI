import keras.backend as K
import tensorflow as tf
import numpy as np

dim0 = 3
dim1 = 4
dim2 = 2
channels = 5
y_true = tf.placeholder(tf.float32, shape=[None, dim0, dim1, dim2, channels])
y_predicted = tf.placeholder(tf.float32, shape=[None, dim0, dim1, dim2, channels])
print y_true


#dim = tf.reduce_prod(tf.shape(y_true)[1:])
#y_true_flatten = tf.reshape(y_true, [-1])
y_true_flatten = K.flatten(y_true)

#y_true_flatten = tf.reshape(y_true, [-1])#,dim_true])
#shape_pred = y_predicted.get_shape().as_list()  # a list: [None, 9, 2]
#dim_pred = np.prod(shape_pred[1:])
#y_pred_flatten = tf.reshape(y_true, [-1])#, dim_pred])
y_pred_flatten = K.flatten(y_predicted)
y_pred_flatten_log = -K.log(y_pred_flatten + K.epsilon())
num_total_elements = K.sum(y_true_flatten)

print "ytrue shape flatten"
print tf.shape(y_true_flatten)
print "ypred shape flatten"
print tf.shape(y_pred_flatten_log)

#cross_entropy = K.dot(y_pred_flatten_log,y_true_flatten)
cross_entropy = tf.reduce_sum(tf.multiply(y_true_flatten, y_pred_flatten_log))
mean_cross_entropy = cross_entropy / (num_total_elements + K.epsilon())

print "I finished!!"
print "cross entropy shape"
print tf.shape(cross_entropy)
print "mean cross entropy shape"
print tf.shape(mean_cross_entropy)
print "mean cross entropy value"
print tf.Print(mean_cross_entropy, [mean_cross_entropy], message="This is mce: ")
