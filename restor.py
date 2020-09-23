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
import cv2
from help_function import help_function as hf
from graph_model import graph_model as gm

heigh_global=56
width_global=56

image_folder = '../Images'
#image_folder = '../oneImage'
image_type = '*.jpg'
batch_size = 1
n_latent = 48  # the size of the code of the bottleneck
inputs_decoder = 3
dec_in_channels = 3  # RGB - 3 Grayscale - 1
n_latent = 48  # the size of the code of the bottleneck

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels / 2

hf = hf()
gm = gm(heigh_global, width_global, n_latent , inputs_decoder, reshaped_dim)

if __name__ == '__main__':



    if not os.path.isdir(image_folder):
        raise Exception('Invalid folder path')

    name_files = []

    batches, no_enter_to_batch = hf.get_batches(image_folder, image_type, batch_size, True)

    hf.extract_details_from_file_name(batches)

    # Training

    X_in, X_detail, Y, loss, img_loss, latent_loss, optimizer, keep_prob, dec, unreshaped = gm.main_graph()

    saver = tf.train.Saver()

    # gender = input("Choose the gender (0 - male, 1 - female): ")
    # years_between_images = input("Choose years range between image and image (5 - 50): ")
    # body_color = input("Choose a body color (0 - 4): ")
    gender = 1

with tf.Session() as sess:

    for i in range(len(batches)):

        batch = batches[i][1]  # Get pixels matrix of current batch from batches list
        details = np.array(batches[i][0])
        details = details.reshape([-1, 7])

        batch_img = [batch[0] for _ in range(1, 101, int(years_between_images))]
        details_img = [[year, gender, 1, 0, 0, 0, 0] for year in range(1, 101, int(years_between_images))]


        # Restore the saved vairable
        saver.restore(sess, '../saver')
        # Print the loaded variable
        imgs = np.reshape(np.array(sess.run([unreshaped], feed_dict={X_in: batch_img, X_detail: details_img, keep_prob: 1.0})), [-1,56,56,3])
        os.mkdir(str(i) + "_" + str(details) + "_VAE")

        for j in range(len(batch_img)):
            im = np.array(imgs[j])
            im = im * 255.0
            f = np.array(im)
            cv2.imwrite(str(i) + "_" + str(details) + "_VAE/" + str(years_between_images * j) + "_1.png", cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            img = smp.toimage(f)
            img.show()