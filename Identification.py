import matplotlib as matplotlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import os
import glob
import math
import scipy.misc as smp
from PIL import Image
import time
import random
from help_function import help_function as hf
from graph_model import graph_model as gm

os.environ["CUDA_VISIBLE_DEVICES"]="0"

heigh_global=56
width_global=56

#image_folder = '../Images'
image_folder = './all_men'
image_type = '*.jpg'
batch_size = 200
inputs_decoder = 3
dec_in_channels = 3  # RGB - 3 Grayscale - 1
save_filename = '../identification_saver/saver_women'
#save_filename = '../identification_women_saver/saver'
train_per = 0.9
hf = hf()


if __name__ == '__main__':



    if not os.path.isdir(image_folder):
        raise Exception('Invalid folder path')

    name_files = []

    train_batches, no_enter_to_batch, test_batch = hf.get_batches(image_folder, image_type, batch_size, True, train_per)

    train_batches = hf.extract_details_from_file_name_women(train_batches)
    test_batch =  hf.extract_details_from_file_name_women(test_batch)

    # train_batches = hf.extract_details_from_file_name(train_batches)
    # test_batch = hf.extract_details_from_file_name(test_batch)

    # Training

    x = tf.placeholder(tf.float32, shape=[None, heigh_global,width_global,3])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    x_image = tf.reshape(x, [-1, heigh_global, width_global, 3])  # if we had RGB, we would have 3 channels

    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 14 * 14 * 64])
    W_fc1 = tf.Variable(tf.truncated_normal([14 * 14 * 64, 1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[1024]))

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # W_fc3 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1))
    # b_fc3 = tf.Variable(tf.constant(0.1, shape=[1024]))
    #
    # h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
    #
    # h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

    W_fc4 = tf.Variable(tf.truncated_normal([1024, 1], stddev=0.1))
    b_fc4 = tf.Variable(tf.constant(0.1, shape=[1]))

    eps = 1e-12

    y_conv = tf.nn.sigmoid(tf.matmul(h_fc2_drop, W_fc4) + b_fc4)
    cross_entropy = tf.reduce_mean(-(y_ * tf.log(y_conv + eps) + (1 - y_) * tf.log( 1 - y_conv + eps)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # uses moving averages momentum
    prediction = tf.round(y_conv)

    correct = tf.cast(tf.equal(prediction, y_), dtype=tf.float32)
    # Average
    accuracy = tf.reduce_mean(correct)

    #my_acc = (tf.equal(y_, predictions), tf.float32)
    #accuracy, _ = tf.metrics.accuracy(y_, predictions)
    #accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    saver.restore(sess, save_filename)

    for i in range(0):
        for j in range(len(train_batches)):
            batch = train_batches[j]
            batch_data_y  = np.array(batch[0])[:, 1].reshape(-1,1)
            if i % 1 == 0 and j == 0:
                #print("cross_entropy: " + str(cross_entropy.eval(feed_dict={x: batch[1], y_: batch_data_y, keep_prob: 1.0})))
                train_accuracy = accuracy.eval(feed_dict={x: batch[1], y_: batch_data_y, keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
                saver.save(sess, save_filename)
            #test_data_y = np.array(batch[0])[:, 1].reshape(-1, 1)
            sess.run(train_step, feed_dict={x: batch[1], y_: batch_data_y, keep_prob: 0.5})
    # test_data_y = np.array(test_batch[0][0])[:, 1].reshape(-1, 1)
    # print("test accuracy %g" % accuracy.eval(feed_dict={x: test_batch[0][1], y_:test_data_y, keep_prob: 1.0}))
    results = prediction.eval(feed_dict={x: test_batch[0][1], keep_prob: 1.0})
    print( results.sum()/ len(results))


    # gender = input("Choose the gender (0 - male, 1 - female): ")
    # years_between_images = input("Choose years range between image and image (5 - 50): ")
    # body_color = input("Choose a body color (0 - 4): ")

