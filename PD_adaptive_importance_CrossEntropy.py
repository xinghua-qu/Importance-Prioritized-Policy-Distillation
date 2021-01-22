import os
import numpy as np
from argparse import ArgumentParser
import pickle
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
from keras import backend as K
from keras.losses import kullback_leibler_divergence,mean_squared_error, categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation, BatchNormalization
from numpy.random import randint
import random
import tensorflow as tf
import argparse
import math
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def CE_loss(y_true, y_pred): 
    softmax_true = K.softmax(y_true)
    softmax_pred = K.softmax(y_pred)
    ce_distance = tf.keras.losses.categorical_crossentropy(softmax_true, softmax_pred)
    return ce_distance


def entropy_calcuclation(X):
    log_X = K.log(X)
    entropy = K.sum(-X*log_X, axis=-1)
    return entropy

def adaptive_entropy_loss(y_train, y_pred, alpha, entropy_temperature,classes): 
    softmax_true = K.softmax(y_train)
    softmax_pred = K.softmax(y_pred)
    ce_distance = tf.keras.losses.categorical_crossentropy(softmax_true, softmax_pred)
    
    ##########
    softmax_train_entropy = K.softmax(y_train/entropy_temperature)
#     softmax_train_entropy = K.clip(softmax_train_entropy, 0, 1)
    entropy_train = entropy_calcuclation(softmax_train_entropy)
    loss_weight = 1/entropy_train    
    loss_weight = K.clip(loss_weight, 0.01, 2)
    
    ##########
    true_label = K.argmax(softmax_true,axis=-1)

    num_examples = tf.cast(tf.shape(y_pred)[0], dtype=true_label.dtype)
    idx = tf.stack([tf.range(num_examples), true_label], axis=-1)
    yg = tf.gather_nd(softmax_pred, idx)
    softmax_pred_entropy = K.softmax(y_pred)
    softmax_pred_entropy = entropy_calcuclation(softmax_pred_entropy)
    entropy_pred = entropy_calcuclation(softmax_pred_entropy)
  
    loss = ce_distance + alpha*yg*entropy_pred
    final_loss = loss*loss_weight
    return final_loss

def main(args):
    saved_data_length = int(5e4)
    train_length = int(9e4)
    test_length  = int(1e4)
    game = args.game
    temp = 1
    ########## Load data #################
    data_dir = '/home/teddy/policy_distillation/policy_distillation/rainbow_dqn_data/'
    with open("{}{}_state_{}_first".format(data_dir,game,saved_data_length), "rb") as fp:
        X_data_1 = pickle.load(fp)
    with open("{}{}_state_{}_second".format(data_dir,game,saved_data_length), "rb") as fp:
        X_data_2 = pickle.load(fp)
    X_data_1 = list(X_data_1)
    X_data_2 = list(X_data_2)
    X = X_data_1 + X_data_2
    X = np.asarray(X)
    with open("{}{}_action_{}_first".format(data_dir,game,saved_data_length), "rb") as fp:
        Y_data_1 = pickle.load(fp)
    with open("{}{}_action_{}_second".format(data_dir,game,saved_data_length), "rb") as fp:
        Y_data_2 = pickle.load(fp)
    Y_data_1 = list(Y_data_1)
    Y_data_2 = list(Y_data_2)
    Y = Y_data_1 + Y_data_2
    Y = np.asarray(Y)
    
    X_train = np.squeeze(X[0:train_length])
    X_test  = np.squeeze(X[train_length:train_length+test_length])
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    X_train = np.swapaxes(X_train, 1, 3)
    X_train = np.swapaxes(X_train, 1, 2)
    X_test  = np.swapaxes(X_test, 1, 3)
    X_test  = np.swapaxes(X_test, 1, 2)
    X_train = X_train/255
    X_test  = X_test/255
    Y_train = np.array(Y[0:train_length])
    Y_test  = np.array(Y[train_length:train_length+test_length])
    classes = len(Y_train[1,:])

    ################ KL loss only ############################
#     model = Sequential()
#     model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=np.shape(X_train[0])))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (4, 4), strides=(2, 2)))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (3, 3), strides=(1, 1)))
#     model.add(Activation('relu'))
#     model.add(Flatten())
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dense(len(Y_train[1,:])))
# #     fn_applied = lambda y_true, y_pred:MSE_loss(y_true, y_pred)

#     adam = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, amsgrad=False)
#     model.compile(optimizer=adam, loss = CE_loss, metrics=[ 'accuracy'])

#     distiled_model_dir = "distiled_model_new/adaptive_entropy_loss_tnnls_CE/{}/".format(game)
#     if not os.path.exists(distiled_model_dir):
#         os.makedirs(distiled_model_dir)
#     save_name = "{}{}_cross_entropy.hdf5".format(distiled_model_dir,game)
#     callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 60),ModelCheckpoint(save_name, save_best_only=True, save_weights_only=False)]
#     model.fit(x=X_train, y=Y_train, validation_data=(X_test, Y_test), batch_size=64, epochs=1501, callbacks= callbacks, use_multiprocessing=True, workers=8)
    
    ################ adaptive entropy loss ############################
    alpha_list = [0.0, 0.5, 1.0]
    entropy_temperature_list = [50, 25, 10, 5, 1, 0.5, 0.1]
    Rewards_adaptive_entropy_loss = []
    for alpha in alpha_list:
        for entropy_temperature in entropy_temperature_list:
            model = Sequential()
            model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=np.shape(X_train[0])))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (4, 4), strides=(2, 2)))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (3, 3), strides=(1, 1)))
            model.add(Activation('relu'))
            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation('relu'))
            model.add(Dense(len(Y_train[1,:])))
            fn_applied = lambda y_true, y_pred:adaptive_entropy_loss(y_true, y_pred, alpha, entropy_temperature, classes)

            adam = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, amsgrad=False)
            model.compile(optimizer=adam, loss = fn_applied, metrics=[ 'accuracy'])

            distiled_model_dir = "distiled_model_new/adaptive_entropy_loss_tnnls_CE/{}/".format(game)
            if not os.path.exists(distiled_model_dir):
                os.makedirs(distiled_model_dir)
            save_name = "{}{}_cross_entropy_alpha_{}_entropytemperature_{}.hdf5".format(distiled_model_dir,game, alpha, entropy_temperature)
            callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 60), ModelCheckpoint(save_name, save_best_only=True, save_weights_only=False)]
            history = model.fit(x=X_train, y=Y_train, validation_data=(X_test, Y_test), batch_size=64, epochs=1501, callbacks= callbacks, use_multiprocessing=True, workers=8)
            ############# save training history 
            history_dir = "{}/history/".format(distiled_model_dir)
            if not os.path.exists(history_dir):
                os.makedirs(history_dir)
            with open('{}training_history_mse_alpha_{}_entropytemperature_{}.pkl'.format(history_dir,alpha, entropy_temperature), 'wb') as file:
                pickle.dump(history.history, file)
                
if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default = '0', help='GPU id')
    parser.add_argument('--game', type=str, default = 'bank_heist', help='the game name')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)