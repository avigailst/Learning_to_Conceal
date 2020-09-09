import tensorflow as tf
from help_function import help_function as hf
hf = hf()
class graph_model:

    def __init__(self,heigh_global, width_global, n_latent, inputs_decoder,reshaped_dim):
        self.heigh_global = heigh_global
        self.width_global = width_global
        self.n_latent = n_latent
        self.inputs_decoder = inputs_decoder
        self.reshaped_dim = reshaped_dim
        return

    def encoder(self, X_in, X_details, keep_prob):

        '''
            The function builds the encoder

        :Args
            :param X_in: batch of pixesl e.g. [ [R1,G1,B1],[R2,G2,B2],[R3,G3,B3] ]
            :param X_details: details of images
            :param keep_prob: dropout probability

        :Returns
            tensor of mean, std, code of bottleneck
        '''
        activation = hf.lrelu
        with tf.variable_scope("vae_encoder", reuse=None):
            X = tf.reshape(X_in, shape=[-1, self.heigh_global, self.width_global, 3], name="vae_encoder_input_reshape")
            x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation, name="vae_encoder_x_1")
            x = tf.nn.dropout(x, keep_prob,name="vae_dropout_x_1")
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="vae_encoder_x_max_pool")
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation, name="vae_encoder_x_2")
            x = tf.nn.dropout(x, keep_prob,name="vae_dropout_x_2")
            x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation, name="vae_encoder_x_3")
            x = tf.nn.dropout(x, keep_prob,name="vae_dropout_x_3")
            x = tf.contrib.layers.flatten(x)
            mn = tf.layers.dense(x, units=self.n_latent, name="vae_encoder_mean")
            sd = 0.5 * tf.layers.dense(x, units=self.n_latent, name="vae_encoder_std")
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent]), name="vae_encoder_epsilon")
            z = mn + tf.multiply(epsilon, tf.exp(sd), name="vae_encoder_z")
            z = tf.concat([z, X_details], 1, name="vae_encoder_z_concut")

            return z, mn, sd

    def decoder(self, sampled_z, keep_prob):
        with tf.variable_scope("decoder", reuse=None):
            x = tf.layers.dense(sampled_z, units=self.inputs_decoder, activation=hf.lrelu, name="vae_decoder_x_1")
            x = tf.layers.dense(x, units=self.inputs_decoder * 2, activation=hf.lrelu, name="vae_decoder_x_2")
            x = tf.reshape(x, self.reshaped_dim)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu, name="vae_decoder_x_3")
            x = tf.nn.dropout(x, keep_prob,name="vae_decoder_dropout_x_1")
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu, name="vae_decoder_x_4")
            x = tf.nn.dropout(x, keep_prob,name="vae_decoder_dropout_x_2")
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu, name="vae_decoder_x_5")

            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=self.heigh_global * self.width_global * 3, activation=tf.nn.sigmoid, name="vae_decoder_x_6")
            img = tf.reshape(x, shape=[-1, self.heigh_global, self.width_global, 3], name="vae_decoder_img")
            return img

    def decoder_with_label(self,sampled_z, keep_prob, label):
        with tf.variable_scope("decoder", reuse=None):
            x = tf.layers.dense(tf.concat([sampled_z, label] , 1) , units=self.inputs_decoder, activation=hf.lrelu, name="vae_decoder_x_1")
            x = tf.layers.dense(x, units=self.inputs_decoder * 2, activation=hf.lrelu, name="vae_decoder_x_2")
            x = tf.reshape(x, self.reshaped_dim)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu, name="vae_decoder_x_3")
            x = tf.nn.dropout(x, keep_prob,name="vae_decoder_dropout_x_1")
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu, name="vae_decoder_x_4")
            x = tf.nn.dropout(x, keep_prob,name="vae_decoder_dropout_x_2")
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu, name="vae_decoder_x_5")

            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=self.heigh_global * self.width_global * 3, activation=tf.nn.sigmoid, name="vae_decoder_x_6")
            img = tf.reshape(x, shape=[-1, self.heigh_global, self.width_global, 3], name="vae_decoder_img")
            return img


    def main_graph(self):


        # Training

        tf.reset_default_graph()

        # batch_size = 64
        X_in = tf.placeholder(dtype=tf.float32, shape=[None, self.heigh_global, self.width_global, 3], name='X')
        X_detail = tf.placeholder(dtype=tf.float32, shape=[None, 7], name='Details')
        Y = tf.placeholder(dtype=tf.float32, shape=[None, self.heigh_global, self.width_global, 3], name='Y')
        Y_flat = tf.reshape(Y, shape=[-1, self.heigh_global * self.width_global * 3])
        keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')




        sampled, mean, std = self.encoder(X_in, X_detail, keep_prob)

        #dec = self.decoder(sampled, keep_prob)
        dec = self.decoder_with_label(sampled , keep_prob, X_detail)
        categories = 1  # for linear

        unreshaped = tf.reshape(dec, [-1, self.heigh_global * self.width_global * 3], name="vae_img_flatten")  # flatten the code
        img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1, name="vae_img_loss")
        latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * std - tf.square(mean) - tf.exp(2.0 * std), 1,
                                           name="vae_latent_loss")

        loss = tf.reduce_mean( img_loss + latent_loss, name="vae_loss")

        tvars = tf.trainable_variables()
        vae_vars = [var for var in tvars if "vae_" in var.name]

        optimizer = tf.train.AdamOptimizer(0.0005, name="vae_Adam").minimize(loss, var_list=vae_vars)

        return (X_in, X_detail, Y, loss, img_loss, latent_loss, optimizer, keep_prob, dec, unreshaped)