# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Dense, Activation, Dropout
from tensorflow.keras.layers import Conv1D, BatchNormalization

tf.keras.backend.set_floatx('float32')


class Conv1dBloc(Layer):
    """Elementary 1D-convolution block for TempCNN encoder"""

    def __init__(self, filters_nb, kernel_size, drop_val, **kwargs):
        super(Conv1dBloc, self).__init__(**kwargs)
        self.conv1D = Conv1D(filters_nb, kernel_size,
                             padding="same", kernel_initializer='he_normal')
        self.batch_norm = BatchNormalization()
        self.act = Activation('relu')
        self.output_ = Dropout(drop_val)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        conv1D = self.conv1D(inputs)
        batch_norm = self.batch_norm(conv1D, training=training)
        act = self.act(batch_norm)
        return self.output_(act, training=training)


class TempCnnEncoder(Layer):
    """"Encoder of SITS on temporal dimension"""

    def __init__(self, drop_val=0.5, **kwargs):
        super(TempCnnEncoder, self).__init__(**kwargs)
        self.conv_bloc1 = Conv1dBloc(64, 5, drop_val)
        self.conv_bloc2 = Conv1dBloc(64, 5, drop_val)
        self.conv_bloc3 = Conv1dBloc(64, 5, drop_val)
        self.flatten = layers.Flatten()

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        conv1 = self.conv_bloc1(inputs, training=training)
        conv2 = self.conv_bloc2(conv1, training=training)
        conv3 = self.conv_bloc3(conv2, training=training)
        flatten = self.flatten(conv3)
        return flatten


class Classifier(Layer):
    """"Generic classifier"""

    def __init__(self, nb_class, nb_units, drop_val=0.5, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.dense = Dense(nb_units)
        self.batch_norm = BatchNormalization()
        self.act = Activation('relu')
        self.dropout = Dropout(drop_val)
        self.output_ = Dense(nb_class, activation="softmax")

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        dense = self.dense(inputs)
        batch_norm = self.batch_norm(dense, training=training)
        act = self.act(batch_norm)
        dropout = self.dropout(act, training=training)
        return self.output_(dropout)


@tf.custom_gradient
def gradient_reverse(x, lamb_da=1.0):
    y = tf.identity(x)

    def custom_grad(dy):
        return lamb_da * -dy, None

    return y, custom_grad


class GradReverse(Layer):
    """"Gradient reversal layer (GRL)"""

    def __init__(self):
        super().__init__()

    def call(self, x, lamb_da=1.0):
        return gradient_reverse(x, lamb_da)


class SpADANN(keras.Model):
    """"SpADANN model is composed of
    a feature extractor: encoder
    a label predictor/classifier: labelClassif
    a domain predictor/classifier: domainClassif
    a GRL to connect feature extractor and domain predictor/classifier
    """

    def __init__(self, nb_class, drop_val=0.5, **kwargs):
        super(SpADANN, self).__init__(**kwargs)
        self.encoder = TempCnnEncoder()
        self.labelClassif = Classifier(nb_class, 256)
        self.grl = GradReverse()
        self.domainClassif = Classifier(2, 256)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False, lamb_da=1.0):
        enc_out = self.encoder(inputs, training=training)
        grl = self.grl(enc_out, lamb_da)
        return enc_out, self.labelClassif(enc_out, training=training), \
               self.domainClassif(grl, training=training)
