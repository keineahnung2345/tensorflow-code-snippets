import tensorflow as tf
import tensorflow.contrib.slim as slim
# https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import numpy as np

height = 299
width = 299
channels = 3

# Create graph
X = tf.placeholder(tf.float32, shape=[None, height, width, channels])
with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2(X, num_classes=1001,is_training=False)
predictions = end_points["Predictions"]
saver = tf.train.Saver()

X_test = np.ones((1,299,299,3))  # a fake image, you can use your own image

# Execute graph
with tf.Session() as sess:
    saver.restore(sess, "./inception_resnet_v2_2016_08_30.ckpt")
    predictions_val = predictions.eval(feed_dict={X: X_test})
    tf.train.write_graph(sess.graph_def, './', 'inception_resnet_v2.pbtxt')
