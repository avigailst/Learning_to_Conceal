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
import copy


class help_function:

    def __init__(self):
        return

    def get_batches(self, folder_path, image_type, size_batch = 200, last_throw = False, train_per = 0.9 ):

        '''
        The function groups the images into a batch, each containing 'batch_size' images

        :Args
            :param folder_path (str): The folder path of the images
            :param image_type (str): The type of the images (format: '*.png', '*.jpg' etc.)
            :param size_batch (int): The size of each batch
            :param last_throw (bool): If throw the last batch if it is not full
        :Returns
            list: List of batches, each batch holds #size_batch names (of files) and #size_batch images (pixels)
            [ [filename1.jpg,filename2.jpg,filename3.jpg],[ [R1,G1,B1],[R2,G2,B2],[R3,G3,B3] ],     [filename4.jpg,]... ]

            int: The number of images in the last batch (if it is not full, if full then the number is 0)
        :Raises
            ValueError: If the value of size_batch is less than 1
        '''

        if size_batch < 1:
            raise ValueError('The size_batch should be greater than 0. The value of size_batch was: {}'.format(size_batch))

        # num_of_images is number files in this folder
        num_of_images = len(next(os.walk(folder_path))[2])

        files_name = os.path.join(folder_path, image_type)

        # glob goes through all files in dir
        images_name = [file_path for file_path in glob.iglob(files_name)] # List of files name
        # images_pixels = mpimg.imread(images_name[0])
        # contains all the pixels of evey file in this dir

        random.seed(2019)
        random.shuffle(images_name)
        images_pixels = [cv2.cvtColor(cv2.imread(file_path, 1),cv2.COLOR_BGR2RGB) for file_path in images_name] # List of pixels # Write "mpimg.imread(file_path)[:,:,0]" for one color
        images_pixels=np.array(images_pixels)
        images_pixels=images_pixels/255.0

        train_batches = []
        train_size = int(train_per*num_of_images)

        for i in range(0, train_size - size_batch, size_batch):
            batch = [images_name[i: i + size_batch],images_pixels[i: i + size_batch]]
            train_batches.append(batch)

        no_enter_to_batch = train_size - len(train_batches)*size_batch

        if((not last_throw and no_enter_to_batch != 0) or no_enter_to_batch == size_batch):
            end_batch_names = images_name[-no_enter_to_batch:]
            end_batch_pixels = images_pixels[-no_enter_to_batch:]
            end_batch = [end_batch_names, end_batch_pixels]
            train_batches.append(end_batch)

            if no_enter_to_batch == size_batch:
                no_enter_to_batch = 0
        test_batch = [images_name[train_size:],images_pixels[train_size:]]

        return train_batches, no_enter_to_batch, [test_batch]

    def extract_details_from_file_name_women(self, batches):

        '''
        The function extract the details of image from the image name (name of file).
        The details contain age, gender and skin color.

        :Args
            :param batches (list): The output from "get_batches" function
        '''

        for batch in batches:
            batch_details = []
            for name in batch[0]:
                try:
                    age_gender_color_by_regex = name.split('_')
                    details = [0] * 7  # Create a list that contains zeros. the number 7 is for [age, gender, color1, color2, color3, color4, color5]
                    #details[0] = int(age_gender_color_by_regex.group(1).replace("_", ""));
                    details[1] = int(age_gender_color_by_regex[2].split('.')[0]);
                    #num_of_color = int(age_gender_color_by_regex.group(3).replace("_", ""));
                    #details[num_of_color + 2] = int(1);
                    batch_details.append(details)
                except:
                    continue
                # img_name = int(age_gender_color_by_regex.group(4).replace("_", ""));
            batch[0] = batch_details
        return batches


    def extract_details_from_file_name(self,batches):

        '''
        The function extract the details of image from the image name (name of file).
        The details contain age, gender and skin color.

        :Args
            :param batches (list): The output from "get_batches" function
        '''

        for batch in batches:
            batch_details = []
            for name in batch[0]:
                try:
                    age_gender_color_by_regex = re.search('([0-9]{1,3})_([0-1])_([0-4])_([0-9]{1,17})', name)
                    details = [0] * 8 # Create a list that contains zeros. the number 7 is for [age, gender, color1, color2, color3, color4, color5]
                    details[0] = int(age_gender_color_by_regex.group(1).replace("_", ""));
                    details[1] = int(age_gender_color_by_regex.group(2).replace("_", ""));
                    num_of_color = int(age_gender_color_by_regex.group(3).replace("_", ""));
                    details[7] = int(age_gender_color_by_regex.group(4).replace("_", ""))
                    details[num_of_color + 2] = int(1);
                    batch_details.append(details)
                except:
                    continue
                #img_name = int(age_gender_color_by_regex.group(4).replace("_", ""));
            batch[0] = batch_details
        return  batches


    def split_train_test(self, full_dataset, train_percent = 0.7, validation_percent = 0.15, test_percent = 0.15):

        '''
        The function splits the list into three lists (train, validation and test) by the given percents.

        :Arg
            :param full_dataset (list): List that want to split
            :param train_percent (double): Number in [0,1]. The percent of information assigned to training
            :param validation_percent (double): Number in [0,1]. The percent of information assigned to validation
            :param test_percent (double): Number in [0,1]. The percent of information assigned to test
        :Returns
            list: three lists representing the train, validation and test
        '''

        # Number of features
        dataset_size = len(full_dataset)

        train_size = math.ceil(train_percent * dataset_size)
        test_validation_size = dataset_size - train_size
        validation_size = math.ceil(validation_percent/(1-train_percent) * test_validation_size)
        test_size = test_validation_size - validation_size

        train_index = train_size
        validation_index = train_index + validation_size
        test_index = dataset_size

        cutting_points = [train_index,validation_index]

        train_data = full_dataset[: train_index]
        validation_data = full_dataset[train_index : validation_index]
        test_data = full_dataset[validation_index :]

        return train_data, validation_data, test_data

    def lrelu(self, x, alpha=0.3):
        return tf.maximum(x, tf.multiply(x, alpha))
