# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:44:56 2021

@author: emmanuel
"""

# importation des modules nécessaires
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Dense, Activation, Dropout
from tensorflow.keras.layers import BatchNormalization

tf.keras.backend.set_floatx('float32')




# *********************************************************************
# Etape de : création du réseau de neurones implémentant DANN
# Domain-Adversarial Neural Network
# sur la base (backbone) du réseau de neurones convolutif temporel TempCNN

# !!! A adapter au cas présent !!!
# ---------------------------------------------------------------------
# Cas d'une architecture à base réseaux de neurones convolutifs pour une classification à 10 classes
# Implémentation ici d'une architecture classique considérée comme fondamentale
# Hyperparamètres du modèle :
# - Nombre de couches de convolution (CONV)
# - Par couche CONV, nombre de filtres, taille du filtre, le pas en hauteur et largeur (par défaut 1x1), gestion des bords et fonction d'activation
# - Nombre de couches d'agrégation (POOL) avec pour chacune la taille de la zone à traiter
# - Enchaînement des couches CC et POOL
# - Nombre de couches de type complètement connectée (CC)
# - Par couche CC, nombre de neurones et fonction d'activation (possibilité de paramètre(s))
# - Nombre de couches de mise à zéro (Dropout) avec pour chacune le ratio d'entrées concernées
# - Fonction d'activation pour la couche de sortie
# - Pour l'étape d'entraînement, la fonction de coût, l'algorithme d'optimisation et éventuellement les métriques à calculer pour mesure la performance
#   la taille d'un lot d'images (batch), par défaut 32, le nombre de fois que l'on apprendra sur le jeu d'entraînement complet (epochs)


# METTRE REF ARTICLE DANN

# **************************************************************
# Création du modèle TempCNN à l'aide de l'API Model Subclassing
# Pelletier, C.; Webb, G.I.; Petitjean, F.
# Temporal Convolutional Neural Network
# for the Classification of Satellite Image Time Series.
# Remote Sens. 2019, 11, 523. https://doi.org/10.3390/rs11050523
# Architecture :
# A COMPLETER


# Définition d'un bloc de convolution 1D
# avec couches de Batch Normalization
class Conv1D_bn_bloc(Layer):

  def __init__(self, filters_nb, kernel_size, drop_val, **kwargs):
    super(Conv1D_bn_bloc, self).__init__(**kwargs) # Appel du constructeur parent

    # Ajout d'un bloc de convolution comprenant :
    # - une couche de convolution avec 'filters_nb' filtres
    #   de taille 'kernel_size' et bord à zéro
    # - une couche de normalisation par lot selon les bandes spectrales
    # - une couche d'activation avec la fonction rectifieur linéaire
    # - une couche de mise à zéro aléatoire avec un taux de 'drop_val'
    #   des valeurs en entrée
    self.conv1D = layers.Conv1D(filters_nb, kernel_size, padding="same", kernel_initializer='he_normal')
    self.batch_norm = BatchNormalization()
    self.act = Activation('relu')
    self.output_ = Dropout(drop_val)

  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False):
    conv1D = self.conv1D(inputs)
    batch_norm = self.batch_norm(conv1D, training=training)
    act = self.act(batch_norm)
    return self.output_(act, training=training)

# Définition d'un bloc de convolution 1D 
# sans couches de Batch Normalization
class Conv1D_bloc(Layer):

  def __init__(self, filters_nb, kernel_size, drop_val, **kwargs):
    super(Conv1D_bloc, self).__init__(**kwargs) # Appel du constructeur parent
     
    # Ajout d'un bloc de convolution comprenant :
    # - une couche de convolution avec 'filters_nb' filtres
    #   de taille 'kernel_size' et bord à zéro
    # - une couche d'activation avec la fonction rectifieur linéaire
    # - une couche de mise à zéro aléatoire avec un taux de 'drop_val'
    #   des valeurs en entrée
    self.conv1D = layers.Conv1D(filters_nb, kernel_size, padding="same",  kernel_initializer='he_normal')
    self.act = Activation('relu')
    self.output_ = Dropout(drop_val)
        
  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False):
    conv1D = self.conv1D(inputs)
    act = self.act(conv1D)
    return self.output_(act, training=training)


# Définition de l'encodeur du TempCNN avec 3 blocs de convolution 1D
# avec couches de Batch Normalization
class TempCNN_BN_Encoder(Layer):

  def __init__(self, drop_val=0.5, **kwargs):
    super(TempCNN_BN_Encoder, self).__init__(**kwargs) # Appel du constructeur parent

    self.conv_bloc1 = Conv1D_bn_bloc(64, 5, drop_val)
    self.conv_bloc2 = Conv1D_bn_bloc(64, 5, drop_val)
    self.conv_bloc3 = Conv1D_bn_bloc(64, 5, drop_val)

    #self.flatten = tf.keras.layers.GlobalAveragePooling1D()
    self.flatten = layers.Flatten()

  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False):

    conv1 = self.conv_bloc1(inputs, training=training)
    conv2 = self.conv_bloc2(conv1, training=training)
    conv3 = self.conv_bloc3(conv2, training=training)

    flatten = self.flatten(conv3)

    return flatten

# Définition de l'encodeur du TempCNN avec 3 blocs de convolution 1D
# sans couches de Batch Normalization
class TempCNN_Encoder(Layer):

  def __init__(self, drop_val=0.5, **kwargs):
    super(TempCNN_Encoder, self).__init__(**kwargs) # Appel du constructeur parent
     
    self.conv_bloc1 = Conv1D_bloc(64, 5, drop_val)
    self.conv_bloc2 = Conv1D_bloc(64, 5, drop_val)
    self.conv_bloc3 = Conv1D_bloc(64, 5, drop_val)
    
    self.flatten = layers.Flatten()
    
  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False):
    
    conv1 = self.conv_bloc1(inputs, training=training)
    conv2 = self.conv_bloc2(conv1, training=training)
    conv3 = self.conv_bloc3(conv2, training=training)

    flatten = self.flatten(conv3)
        
    return flatten


# Définition d'un classifieur pour nb_class classes
# avec une couche complètement connectée de nb_units neurones
# avec couches de Batch Normalization
class Classifier_BN(Layer):

  def __init__(self, nb_class, nb_units, drop_val=0.5, **kwargs):
    super(Classifier_BN, self).__init__(**kwargs) # Appel du constructeur parent

    self.dense = Dense(nb_units)
    self.batch_norm = BatchNormalization()    
    self.act = Activation('relu')
    self.dropout = Dropout(drop_val)

    # couche de nb_class neurones, 1 sortie par classe, la valeur de chaque sortie
    # = probabilité estimée que l'entrée corresponde ait cette classe
    # (fonction softmax)
    self.output_ = Dense(nb_class, activation="softmax")

  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False):

    dense = self.dense(inputs)
    batch_norm = self.batch_norm(dense, training=training)
    act = self.act(batch_norm)
    dropout = self.dropout(act, training=training)
    return self.output_(dropout)

# Définition d'un classifieur pour nb_class classes
# avec une couche complètement connectée de nb_units neurones
# sans couches de Batch Normalization
class Classifier(Layer):

  def __init__(self, nb_class, nb_units, drop_val=0.5, **kwargs):
    super(Classifier, self).__init__(**kwargs) # Appel du constructeur parent
     
    self.dense = Dense(nb_units)
    self.act = Activation('relu')
    self.dropout = Dropout(drop_val)     

    # couche de nb_class neurones, 1 sortie par classe, la valeur de chaque sortie
    # = probabilité estimée que l'entrée corresponde ait cette classe
    # (fonction softmax)
    self.output_ = Dense(nb_class, activation="softmax")
  
  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False):    
    dense = self.dense(inputs)
    act = self.act(dense)
    dropout = self.dropout(act, training=training)
    return self.output_(dropout)


@tf.custom_gradient
def gradient_reverse(x, lamb_da=1.0):
    y = tf.identity(x)
    def custom_grad(dy):
        return lamb_da * -dy, None
    return y, custom_grad


class GradReverse(Layer):
    def __init__(self):
        super().__init__()

    def call(self, x, lamb_da=1.0):
        return gradient_reverse(x, lamb_da)


# Définition du modèle DANN
# avec couches de Batch Normalization
class DANN_BN_Model(keras.Model):
  
  def __init__(self, nb_class, drop_val=0.5, **kwargs):
    super(DANN_BN_Model, self).__init__(**kwargs) # Appel du constructeur parent
    
    # Feature Extractor
    self.encoder = TempCNN_BN_Encoder()
      
    # Label Predictor/Classifier    
    self.labelClassif = Classifier_BN(nb_class, 256)
            
    self.grl = GradReverse()
    
    # Domain Predictor/Classifier    
    self.domainClassif = Classifier_BN(2, 256)

  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False, lamb_da=1.0):
    
    enc_out = self.encoder(inputs, training=training)
    grl = self.grl(enc_out, lamb_da)
    return enc_out, self.labelClassif(enc_out, training=training),\
        self.domainClassif(grl, training=training)


# Définition du modèle DANN
# sans couches de Batch Normalization
class DANN_Model(keras.Model):
  
  def __init__(self, nb_class, drop_val=0.5, **kwargs):
    super(DANN_Model, self).__init__(**kwargs) # Appel du constructeur parent

    # Feature Extractor
    self.encoder = TempCNN_Encoder()
    
    # Label Predictor/Classifier    
    self.labelClassif = Classifier(nb_class, 256)
        
    self.grl = GradReverse()
    
    # Domain Predictor/Classifier    
    self.domainClassif = Classifier(2, 256)

  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False, lamb_da=1.0):
    
    enc_out = self.encoder(inputs, training=training)
    grl = self.grl(enc_out, lamb_da)
    return enc_out, self.labelClassif(enc_out, training=training),\
        self.domainClassif(grl, training=training)
