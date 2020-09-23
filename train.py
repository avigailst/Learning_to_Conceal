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
import random
import tensorflow as tf
from help_function import help_function as hf
from graph_model import graph_model as gm

show = True
iteration_number =0
to_restor = True
ifGender = True
heigh_global=56
width_global=56
image_folder = '../Images'
image_type = '*.jpg'
batch_size = 100

inputs_decoder = 3
dec_in_channels = 3  # RGB - 3 Grayscale - 1
n_latent = 2500  # the size of the code of the bottleneck
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
    data = hf.get_data(image_folder, image_type)
    data = hf.extract_details_from_file_name_two_color(data)
    #batches, no_enter_to_batch, test_batches = hf.get_batches( data, batch_size, True, train_per)


    X_in, X_detail,X_detail_new,Y, Y_flat, loss, img_loss, latent_loss, sampled_loss,img_loss_2, optimizer, keep_prob, dec, unreshaped_1, unreshaped_2 = gm.main_graph()

    saver = tf.train.Saver()
    saver_path = './saver_new/saver'

    with tf.Session() as sess:

        # initialize all of the variables in the session
        sess.run(tf.global_variables_initializer())
        if(to_restor):
            saver.restore(sess, saver_path)
        min_loss = 700


        k= 0
        for i in range(iteration_number):

            k += 1
            batch =  hf.get_next_batch(data, k)
            if batch is None:
                k=0
                continue

              # Get pixels matrix of current batch from batches list
            #batch = batch.reshape([-1, 56, 56, 3])

            batch = np.array(batch)
            details_all = batch[ :, 0]

            details = np.array([line[:7] for line in details_all])

            data_img = np.array([line[1] for line in batch])


            detail_new = details.copy()
            if ifGender:
                detail_new[:, 1] = 1-detail_new[:, 1]
            else:
                detail_new[:, 2] = 1-detail_new[:, 2]
                detail_new[:, 3] = 1-detail_new[:,3]

            sess.run(optimizer, feed_dict={X_in: data_img, X_detail: details, X_detail_new : detail_new, Y: data_img, keep_prob: 0.8})

            if i % 5 == 0:

                ls, i_ls, d_ls, i_ls_2, s_los = sess.run([loss, img_loss, latent_loss, img_loss_2,sampled_loss], feed_dict={X_in: data_img, X_detail: details, X_detail_new : detail_new, Y: data_img, keep_prob: 1.0})


                print('Iteration:', i, ' loss:', ls, 'img_loss', np.mean(i_ls), "image_loss_2", np.mean(i_ls_2), "sample_loss", np.mean(s_los))


            saver.save(sess, saver_path)




        # local_loss = loss.eval(feed_dict={X_in: batch, X_detail: details,X_detail_new : detail_new, Y: batch, keep_prob: 1.0})


        if show:
            saver.restore(sess, saver_path)
            for i in range(10):
                batch = hf.get_next_batch(data, 1)

                oneImg = batch[i]

                details = np.array(oneImg[0][:7])
                data_img = np.array(oneImg[1])

                detail_new = details.copy()
                if ifGender:
                    detail_new[1] = 1 - detail_new[1]
                else:
                    detail_new[ 2] = 1 - detail_new[2]
                    detail_new[3] = 1 - detail_new[3]
                (imgs1,imgs2, imgs3) = np.reshape(
                    np.array(sess.run([Y_flat, unreshaped_1,unreshaped_2  ], feed_dict={X_in: [data_img], X_detail: [details], X_detail_new : [detail_new], Y: [data_img], keep_prob: 1.0})),
                    [-1, 56, 56, 3])
                jj= 0
                for img in (imgs1,imgs2, imgs3):
                    im = np.array(img)
                    im = im * 255.0
                    cv2.imwrite("./images_restor/"+ str(jj) +"_" + str(details) + ".jpg",
                        cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    jj+=1

