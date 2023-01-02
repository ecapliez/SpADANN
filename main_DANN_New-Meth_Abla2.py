# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:30:00 2022

@author: emmanuel
"""

# importation des modules nécessaires
import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from scipy.spatial.distance import cdist

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from DANN_nn_models import DANN_BN_Model, DANN_Model

tf.keras.backend.set_floatx('float32')

# Calculs sur le GPU dont l'index est passé en 8 paramètres
if sys.argv[8] == "-1":
    # Computation on CPU only, no use of GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
else:
    # Computation on GPU with index sys.argv[8]
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[8]
    # Empêche l'allocation de toute la mémoire du GPU, alloue le juste besoin
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)


# Déclaration et définition des variables globales
loss_pred_set = []
loss_domain_set = []
loss_pred_set_PL = []
loss_combined_set = []
target_f1 = []
PL_accuracy = []


def print_object_info(obj):
    print(type(obj))
    print(obj.shape)
    print(obj.dtype)
    print(obj)
    

def getBatch(X, i, batch_size):
    start_id = (i*batch_size)
    end_id = min((i+1) * batch_size, X.shape[0])
    batch_x = X[start_id:end_id]
    return batch_x


def trainingDANNModel(model, optimizer, s_X, s_y, t_X, test_X, test_y, 
                      num_epochs, batch_size, bn_flag, loss_fn, nb_class):
    
    global loss_pred_set, loss_domain_source_set, loss_domain_target_set
    global loss_pred_set_PL, loss_combined_set, target_f1, PL_accuracy
    global beta
    
    epochs = range(num_epochs)
    
    nb_samples = s_X.shape[0]
    iterations = nb_samples / batch_size
        
    if nb_samples % batch_size != 0:
        iterations += 1
    
    for epoch in epochs:
        t_y = test_y.copy()
        s_X, s_y, t_X, t_y = shuffle(s_X, s_y, t_X, t_y) 
                
        alpha = beta * (float(epoch) / num_epochs)
        lamb_da = 2 / (1 + np.exp(-10 * (float(epoch) / num_epochs),
                                  dtype=np.float32)) - 1
        lamb_da = lamb_da.astype('float32')
        
        PL_counter = 0
        PL_global_set = []
        test_y_2cpe = []
                                
        for ibatch in range(int(iterations)):
            batch_source = getBatch(s_X, ibatch, batch_size)
            batch_target = getBatch(t_X, ibatch, batch_size)
            batch_y = getBatch(s_y, ibatch, batch_size)
            batch_y_target = getBatch(t_y, ibatch, batch_size)
                        
            with tf.GradientTape() as tape:
                emb_source, lpred_source, dpred_source = model(batch_source,
                                                        bn_flag, lamb_da)
                emb_target, lpred_target, dpred_target = model(batch_target,
                                                               bn_flag,
                                                               lamb_da)
                                              
                loss_pred = loss_fn(batch_y, lpred_source)
                loss_domain\
                    = loss_fn(tf.concat([np.ones(batch_source.shape[0]), 
                                         np.zeros(batch_target.shape[0])], 
                                        axis=0),
                              tf.concat([dpred_source,dpred_target], 
                                        axis=0))
                
                if epoch == 0:
                    loss_combined = loss_pred + loss_domain
                else:
                    lpred_src = np.argmax(lpred_source, axis=1)
                    lpred_tgt = np.argmax(lpred_target, axis=1)
                                                      
                    first_cond = (lpred_src == batch_y)
                    second_cond = (lpred_src == lpred_tgt)
                    result\
                        = (first_cond & second_cond).astype(int)
                    loss_pred_PL\
                        = loss_fn(lpred_tgt, lpred_target,
                                  sample_weight=result)
                    loss_combined = (1 - alpha)\
                        * (loss_pred + loss_domain) + alpha * loss_pred_PL
                    
                    PL_global_set.append(lpred_tgt[np.where(result == 1)])
                    test_y_2cpe.append(batch_y_target[np.where(result == 1)])
                    PL_counter += result.sum()
                                            
            grads = tape.gradient(loss_combined, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            train_loss(loss_pred)
            domain_loss(loss_domain)
            if 'loss_pred_PL' in locals():
                train_loss_PL(loss_pred_PL)
            value_loss(loss_combined)
                
        _, pred_test_target, _ = model.predict(test_X, batch_size=1024)
                
        fscoreT = f1_score(test_y, np.argmax(pred_test_target, axis=1),
                           average="weighted")
        
        if epoch == 0:
            print("Epoch %d | TRAIN LOSS %.5f | DOMAIN LOSS %.5f |"
                  " TRAIN+DOMAIN LOSS %.5f | TARGET F1-score %.3f"
                  % (epoch, train_loss.result(), domain_loss.result(),
                     value_loss.result(), fscoreT))
        else:
            PL_global_set\
                = np.concatenate(np.asarray(PL_global_set, dtype=object), axis=0)
            test_y_2cpe\
                = np.concatenate(np.asarray(test_y_2cpe, dtype=object), axis=0)
            accuracy_PL = accuracy_score(test_y_2cpe, PL_global_set)
            print("Epoch %d | TRAIN LOSS %.5f | DOMAIN LOSS %.5f |"
                  " PL_LOSS %.5f | TRAIN+DOMAIN+PL LOSS %.5f "
                  "| TARGET F1-score %.3f | #PL %d | PL ACC %.3f"
                  % (epoch, train_loss.result(), domain_loss.result(),
                     train_loss_PL.result(), value_loss.result(),
                     fscoreT, PL_counter, accuracy_PL))
            PL_accuracy.append(accuracy_PL)
      
        # Sauvegarde des scores dans des tableaux 1D
        loss_pred_set.append(train_loss.result())
        loss_domain_set.append(domain_loss.result())
        loss_pred_set_PL.append(train_loss_PL.result())
        loss_combined_set.append(value_loss.result())
        target_f1.append(fscoreT)
                    
        
# ################################
# # Corps principal du programme :

# Test du nombre d'arguments en entrée du script
if len(sys.argv) != 9:
    print("!!! Erreur : Nombre d'arguments au script 'main.py' incorrect !!!")
    print("!!! Arrêt du script                                           !!!")
    exit()

# Récupération des valeurs passées en argument du script
# Argument 1 : année du jeu de données du domaine source
# Argument 2 : année du jeu de données du domaine cible
# Argument 3 : numéro de configuration du modèle à utiliser
# Argument 4 : learning rate
# Argument 5 : number of epochs
# Argument 6 : batch size
# Argument 7 : n° GPU en partant de 0
s_year = sys.argv[1]
t_year = sys.argv[2]
config_num = int(sys.argv[3])
learning_rate = float(sys.argv[4])
num_epochs = int(sys.argv[5])
batch_size = int(sys.argv[6])
beta = float(sys.argv[7])

start_date = datetime.now()
print("Calculs débutés à la date du : %s" % str(start_date))

# Déclaration et définition des constantes

# Chemin vers le répertoire contenant les sauvegardes des meilleurs modèles
pth2MD = "Model_Data/"

# Chemin vers le répertoire contenant les sauvegardes des meilleurs modèles
pth2R = "Results/"

# Chemin vers le répertoire contenant les sauvegardes *.npy des jeux de données
pth2D_Y1 = "Data/"+s_year+"/"
pth2D_Y2 = "Data/"+t_year+"/"

suffix = '_'+sys.argv[1]+'_'+sys.argv[2]+'_'\
    +sys.argv[3]+'_'+sys.argv[4]+'_'\
        +sys.argv[5]+'_'+sys.argv[6]+'_'+sys.argv[7]

# Chargement des jeux de données d'observation et de vérité terrain
s_X = np.load(pth2D_Y1+'full_X.npy').astype('float32')
t_X = np.load(pth2D_Y2+'full_X.npy').astype('float32')
s_y = np.load(pth2D_Y1+'full_y.npy').astype('uint8') - 1
t_y = np.load(pth2D_Y2+'full_y.npy').astype('uint8') - 1

# full_x_y = np.load(pth2D+'full_x_y_Y1.npy')

nb_class = len(np.unique(s_y))


# Instanciation du modèle du numéro de sa configuration
print("Choix de la configuration du framework DANN")
if config_num == 1:
    model = DANN_BN_Model(nb_class)
    model_file_name = "SpADANN_BN_Abla2"
    sd = model_file_name + suffix + "/"
    bn_flag = True
elif config_num == 2:
    model = DANN_BN_Model(nb_class)
    model_file_name = "best_source_DANN_BN"
    sd = model_file_name + suffix + "/"
    bn_flag = False
elif config_num == 3:
    model = DANN_Model(nb_class)
    model_file_name = "best_source_DANN"
    sd = model_file_name + suffix + "/"
    bn_flag = True
else:
    print("!!! Erreur : numéro de configuration du modèle inconnu !!!")
    print("!!! Arrêt du script                                     !!!")
    exit()


# *********************************************************************
# Etape de : 

# Instanciation d'une fonction objet à partir du modèle de l'entropie croisée
# pour des classes éparses et exclusives
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# Instanciation d'un optimiseur 
optimizer = keras.optimizers.Adam(learning_rate)

# Définition des métriques mesurées : pertes et précisions
train_loss = tf.keras.metrics.Mean(name='train_loss')
domain_loss = tf.keras.metrics.Mean(name='domain_loss')
train_loss_PL = tf.keras.metrics.Mean(name='train_loss_PL')
value_loss = tf.keras.metrics.Mean(name='value_loss')


print("###########################")
print("Début boucle d'entraînement")
print("---------------------------")

trainingDANNModel(model, optimizer, s_X, s_y, t_X, t_X, t_y, 
                  num_epochs, batch_size, bn_flag, loss_fn, nb_class)

# Sauvegarde des paramètres du modèle appris
model.save_weights(pth2MD+sd+"best_"+model_file_name)

print("Fin boucle d'entraînement")
print("#########################")
    
_, pred_test_target, _ = model.predict(t_X, batch_size=1024)
       
# Sauvegarde des prédictions et de la vérité terrain
# suite évaluation sur le domaine cible
Path(pth2R+sd).mkdir(exist_ok=True, parents=True)
np.save(pth2R+sd+"best_"+model_file_name+"-t_X", t_X)
np.save(pth2R+sd+"best_"+model_file_name+"-t_y", t_y)
np.save(pth2R+sd+"best_"+model_file_name+"-predictions", pred_test_target)

y_pred = np.argmax(pred_test_target, axis=1)

accuracy_TD = np.round(accuracy_score(t_y, y_pred), 3)
f1_score_TD = np.round(f1_score(t_y, y_pred, average='weighted'), 3)
kappa_TD = np.round(cohen_kappa_score(t_y, y_pred), 3)

print('\n')
print('**********************************************************************')
print("Résultats finaux : Accuracy =%1.3f" % accuracy_TD)
print('**********************************************************************')
print("Résultats finaux : F1 score (average='weighted') =%1.3f" % f1_score_TD)
print('**********************************************************************')
print("Résultats finaux : Cohen's Kappa score =%1.3f" % kappa_TD)
print('**********************************************************************')
print('\n')

   
# Génération du graphique de résultats uniquement pour le test sur folder 5
x_axis = [j for j in range(0, num_epochs)]
x_axis_PL = [j for j in range(1, num_epochs)]

plt.plot(x_axis, loss_pred_set, label="loss_train")
plt.plot(x_axis, loss_domain_set, label="loss_domain")
plt.plot(x_axis, loss_pred_set_PL, label="loss_train_PL")
plt.plot(x_axis, loss_combined_set, label="loss_combined")
plt.plot(x_axis, target_f1, label="f1_target")
plt.plot(x_axis_PL, PL_accuracy, label="accuracy_PL")
plt.legend()
plt.savefig(pth2R + model_file_name+suffix + '.png')
plt.close()
    

# Fin de l'étape
# **************

print("Déroulement OK !")

end_date = datetime.now()
print("Calculs terminés à la date du : %s" % str(end_date))
print("Durée des calculs : %s" % str(end_date - start_date))
