import matplotlib as matplotlib
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
import cv2
import copy
import tensorflow as tf
from help_function import help_function as hf
from graph_model import graph_model as gm


heigh_global=56
width_global=56
image_folder = '../Images'
image_type = '*.jpg'
batch_size = 100
n_latent = 100  # the size of the code of the bottleneck
inputs_decoder = 3
dec_in_channels = 3  # RGB - 3 Grayscale - 1
n_latent = 48  # the size of the code of the bottleneck
train_per = 0.8
reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels / 2

hf = hf()
gm = gm(heigh_global, width_global, n_latent , inputs_decoder, reshaped_dim)


if __name__ == '__main__':

    # Read and analyze dataset




    if not os.path.isdir(image_folder):
        raise Exception('Invalid folder path')

    name_files = []


    batches, no_enter_to_batch, test_batches = hf.get_batches(image_folder, image_type, batch_size, True, train_per)

    random.seed(2019)
    random.shuffle(batches)

    hf.extract_details_from_file_name(batches)

    X_in, X_detail, Y, loss, img_loss, latent_loss, optimizer, keep_prob, dec, _ = gm.main_graph()

    saver = tf.train.Saver()
    saver_path = '../saver_complex'

    with tf.Session() as sess:

        # initialize all of the variables in the session
        sess.run(tf.global_variables_initializer())

        min_loss = 700

        k = 0

        for i in range(10000):
            for oneBatch in batches:
                batch = np.array(oneBatch[1])  # Get pixels matrix of current batch from batches list
                batch = batch.reshape([-1, 56, 56, 3])
                details = np.array(oneBatch[0])
                details = details.reshape([-1, 8])
                details = details[:, :7]

                sess.run(optimizer, feed_dict={X_in: batch, X_detail: details, Y: batch, keep_prob: 0.8})
                imgs = sess.run(dec, feed_dict={X_in: [batch[0]], X_detail: [details[0]], keep_prob: 1.0})

                ls, i_ls, d_ls = sess.run([loss, img_loss, latent_loss],
                                          feed_dict={X_in: batch, X_detail: details, Y: batch,
                                                     keep_prob: 1.0})

                if k % 100 == 0:

                    ls, i_ls, d_ls = sess.run([loss, img_loss, latent_loss], feed_dict={X_in: batch, X_detail: details, Y: batch, keep_prob: 1.0})

                    imgs = sess.run(dec, feed_dict={X_in: [batch[0]], X_detail: [details[0]], keep_prob: 1.0})
                    end = time.time()


                    local_loss = loss.eval(feed_dict={X_in: batch, X_detail: details, Y: batch, keep_prob: 1.0})
                    if min_loss > local_loss * 1.1 and local_loss < 150:
                        min_loss = local_loss
                        saver.save(sess, saver_path)





            k += 1

            local_loss = loss.eval(feed_dict={X_in: batch, X_detail: details, Y: batch, keep_prob: 1.0})
            print('Iteration:', i, ' loss:', local_loss)

