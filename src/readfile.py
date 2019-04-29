# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf

def write_file():
    v1 = tf.Variable(tf.random_normal([1, 2]), dtype=tf.float32, name='v1')
    v2 = tf.Variable(tf.random_normal([2, 3]), dtype=tf.float32, name='v2')
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

    with tf.Session() as sess:
        sess.run(init_op)
        print("v1:", sess.run(v1))  # 打印v1、v2的值一会读取之后对比
        print("v2:", sess.run(v2))
        saver_path = saver.save(sess, "save/model.ckpt")  # 将模型保存到save/model.ckpt文件
        print("Model saved in file:", saver_path)


def read_file():
    saver = tf.train.import_meta_graph("vgg16/model.ckpt.meta")
    with tf.Session() as sess:
        saver.restore(sess, "vgg16/model.ckpt")
        #print(sess.run(tf.get_default_graph().get_tensor_by_name("v2:0")))
        print(sess.run(tf.get_default_graph().get_all_collection_keys()))


#write_file()
read_file()
