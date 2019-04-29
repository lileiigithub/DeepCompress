__author__ = 'LiLei'
from compress import Compress
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import time

path = "C:/MyCode_/python/mnist/save/model.ckpt"
reader = pywrap_tensorflow.NewCheckpointReader(path)
w1 = reader.get_tensor("w1")
b1 = reader.get_tensor("b1")
w2 = reader.get_tensor("w2")
w3 = reader.get_tensor("w3")
b2 = reader.get_tensor("b2")
b3 = reader.get_tensor("b3")
w4 = reader.get_tensor("w4")
b4 = reader.get_tensor("b4")
#print(w1)
cm = Compress()

target_layers=[(w1,b1),(w2,b2),(w3,b3),(w4,b4)]  #,
for layer in target_layers:
    print(layer[0].shape,layer[1].shape)

start_time = time.time()

# the biggest cluster_num is 256-1,beacuese it's labels stored in 1Byte.
cm.compress_network(target_layers,0.2,256-1,"save_compressed/mnist_fastercluster_L2_huffman")
end_time = time.time()
print("used time:",end_time-start_time)

'''
W1 = tf.Variable(w1)
W2 = tf.Variable(w2)
W3 = tf.Variable(w3)
W4 = tf.Variable(w4)
B1 = tf.Variable(b1)
B2 = tf.Variable(b2)
B3 = tf.Variable(b3)
B4 = tf.Variable(b4)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver([W1,B1,W2,B2,W4,B4])
    saver.save(sess,"save_compressed/vgg_16")
'''

'''
target_layers_=[(5,5,1,32),(5,5,32,64),(1024, 10)]
value = cm.decode_network("save_compressed/mnist_0.2_255_clusters.npy",target_layers_)
print(value)
'''
