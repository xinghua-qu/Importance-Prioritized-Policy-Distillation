from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras import backend as K
from keras.losses import kullback_leibler_divergence,mean_squared_error
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation, BatchNormalization
from numpy.random import randint
import random
import tensorflow as tf

from keras import Model
from keras import Input
from keras.models import load_model
from art.attacks import ProjectedGradientDescent, CarliniL2Method, SaliencyMapMethod
from art.classifiers import KerasClassifier
import keras.losses

import torch
import argparse
from env import Env
import atari_py

from agent import Agent
import os
import numpy as np
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Restrict TensorFlow to only use the fourth GPU
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(10e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
parser.add_argument('--lb', type=int, default=20,  help='lower bound of reward for collect data')

# Setup
args = parser.parse_args()
# args.evaluation_episodes = 100
# args.model = 'teacher_model/{}.pth'.format(args.game)
# args.memory = 'teacher_model/{}.pth'.format(args.game)
args.device = torch.device('cpu')
# args.hidden_size = 512

# game_list   =  ['bank_heist','pong', 'boxing','road_runner','freeway','breakout','qbert']
game = args.game
game_list = [args.game]
evaluation_times = 15

################ CE loss only ############################
# tau_list = [0.01, 0.1, 1, 10]
# for tau in tau_list:
#     if tau>0.5:
#         tau = int(tau)
args.id = args.game
args.evaluation_size = 1000
args.seed = int(123)
env = Env(args)
env.eval()
action_space = env.action_space()
logfile = open('results/TNNLS/{}_CE_loss_evaluation_new.txt'.format(game), 'a+')
head = '%%%%%%%%%%%%%%% game = {}  >>>>>> loss = CE %%%%%%%%%%%% \n'.format(game)
logfile.write(head)
# create model structure
model = Sequential()
model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=(84,84,4)))
model.add(Activation('relu'))
model.add(Conv2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(action_space))
model.summary()   
TEMP_RESULT_LIST = []  
# load weights into model
distiled_model_dir = "distiled_model_new/adaptive_entropy_loss_tnnls_CE/{}/".format(game)
save_name = "{}{}_cross_entropy.hdf5".format(distiled_model_dir,game)
model.load_weights("{}".format(save_name))
classifier    = KerasClassifier(model=model)
pgd_attack_4  = ProjectedGradientDescent(classifier=classifier, eps=0.004, eps_step=0.001, max_iter=4)

### no attack
done = True
Clean_reward = []
for j in range(evaluation_times):
    while True:
        if done:
            state, reward_sum, done = env.reset(), 0, False
        state = np.asarray(state.tolist())
        state = np.expand_dims(state, axis=0)
        state = np.swapaxes(state, 1, 3)
        state = np.swapaxes(state, 1, 2)
        actions = model.predict(state)
        action = np.argmax(actions,axis=1)
        state, reward, done = env.step(action[0])  # Step
        reward_sum += reward
        if done: 
            print(reward_sum)
            Clean_reward.append(reward_sum)
            break
print('  CE loss\n')
print('       {}\n'.format(Clean_reward))
print('       No attack Reward_mean = {}; Reward_std = {}\n'.format(np.mean(Clean_reward), np.std(Clean_reward)))
record = '   CE loss \n'
logfile.write(record)
record = '       {}\n'.format(Clean_reward)
logfile.write(record)
record = '       No attack Reward_mean = {}; Reward_std = {}\n'.format(np.mean(Clean_reward), np.std(Clean_reward))
logfile.write(record)

### PGD-4 attack
Reward_pgd_attack_4 = []
for j in range(evaluation_times):
    while True:
        if done:
            state, reward_sum, done = env.reset(), 0, False
        state = np.asarray(state.tolist())
        state = np.expand_dims(state, axis=0)
        state = np.swapaxes(state, 1, 3)
        state = np.swapaxes(state, 1, 2)
        state = pgd_attack_4.generate(x=state.astype(np.float32))
        actions = model.predict(state)
        action = np.argmax(actions,axis=1)            
        state, reward, done = env.step(action[0])  # Step
        reward_sum += reward
        if done:  
            print(reward_sum)
            Reward_pgd_attack_4.append(reward_sum)
            break
print('       {}\n'.format(Reward_pgd_attack_4))
print('       PGD-4 Reward_mean = {}; Reward_std = {}\n'.format(np.mean(Reward_pgd_attack_4), np.std(Reward_pgd_attack_4)))
record = '       {}\n'.format(Reward_pgd_attack_4)
logfile.write(record)
record = '       PGD-4 Reward_mean = {}; Reward_std = {}\n'.format(np.mean(Reward_pgd_attack_4), np.std(Reward_pgd_attack_4))
logfile.write(record)
logfile.close()
###############################################################################
###############################################################################


################ adaptive entropy loss ############################
alpha_list = [0.0, 0.5, 1.0]
entropy_temperature_list = [50, 25, 10, 5, 1, 0.5, 0.1]
# alpha_list = [0.0, 0.5, 1.0, 5]
# entropy_temperature_list = [5.0, 1.0, 0.75, 0.5, 0.25, 0.05, 0.01]
RESULTS_LIST = []
args.id = args.game
args.evaluation_size = 1000
args.seed = int(123)
env = Env(args)
env.eval()
action_space = env.action_space()
tau = 0.01
logfile = open('results/TNNLS/CE_{}_adaptive_importance_distillation_evaluation_tnnls.txt'.format(game), 'a+')
head = '%%%%%%%%%%%%%%% game = {} %%%%%%%%%%%% \n'.format(game)
logfile.write(head)
for alpha in alpha_list:
    CleanMean_alpha = []
    CleanStd_alpha  = []
    AdvMean_alpha   = []
    AdvStd_alpha    = []
    for entropy_temperature in entropy_temperature_list:                
        # create model structure
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=(84,84,4)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(action_space))
        model.summary()   
        # load weights into model
        distiled_model_dir = "distiled_model_new/adaptive_entropy_loss_tnnls_CE/{}/".format(game)
        save_name = "{}{}_cross_entropy_alpha_{}_entropytemperature_{}.hdf5".format(distiled_model_dir,game, alpha, entropy_temperature)
        model.load_weights("{}".format(save_name))
        classifier    = KerasClassifier(model=model)
        pgd_attack_4  = ProjectedGradientDescent(classifier=classifier, eps=0.004, eps_step=0.001, max_iter=4)
        
        logfile = open('results/TNNLS/CE_{}_adaptive_importance_distillation_evaluation_tnnls.txt'.format(game), 'a+')
        ### no attack
        done = True
        Clean_reward = []
        for j in range(evaluation_times):
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                state = np.asarray(state.tolist())
                state = np.expand_dims(state, axis=0)
                state = np.swapaxes(state, 1, 3)
                state = np.swapaxes(state, 1, 2)
                actions = model.predict(state)
                action = np.argmax(actions,axis=1)
                state, reward, done = env.step(action[0])  # Step
                reward_sum += reward
                if done: 
                    print(reward_sum)
                    Clean_reward.append(reward_sum)
                    break
        print('  kl = {}; alpha = {}; entropy_temperature = {}\n'.format(tau, alpha, entropy_temperature))
        print('       {}\n'.format(Clean_reward))
        print('       No attack Reward_mean = {}; Reward_std = {}\n'.format(np.mean(Clean_reward), np.std(Clean_reward)))
        record = '  kl = {}; alpha = {}; entropy_temperature = {}\n'.format(tau, alpha, entropy_temperature)
        logfile.write(record)
        record = '       {}\n'.format(Clean_reward)
        logfile.write(record)
        record = '       No attack Reward_mean = {}; Reward_std = {}\n'.format(np.mean(Clean_reward), np.std(Clean_reward))
        logfile.write(record)
        CleanMean_alpha.append(np.mean(Clean_reward))
        CleanStd_alpha.append(np.std(Clean_reward))

        ### PGD-4 attack
        Reward_pgd_attack_4 = []
        for j in range(evaluation_times):
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                state = np.asarray(state.tolist())
                state = np.expand_dims(state, axis=0)
                state = np.swapaxes(state, 1, 3)
                state = np.swapaxes(state, 1, 2)
                state = pgd_attack_4.generate(x=state.astype(np.float32))
                actions = model.predict(state)
                action = np.argmax(actions,axis=1)            
                state, reward, done = env.step(action[0])  # Step
                reward_sum += reward
                if done:  
                    print(reward_sum)
                    Reward_pgd_attack_4.append(reward_sum)
                    break
        print('       {}'.format(Reward_pgd_attack_4))
        print('       PGD-4 Reward_mean = {}; Reward_std = {}\n'.format(np.mean(Reward_pgd_attack_4), np.std(Reward_pgd_attack_4)))
        record = '       {}\n'.format(Reward_pgd_attack_4)
        logfile.write(record)
        record = '       PGD-4 Reward_mean = {}; Reward_std = {}\n'.format(np.mean(Reward_pgd_attack_4), np.std(Reward_pgd_attack_4))
        logfile.write(record)
        logfile.close()
        AdvMean_alpha.append(np.mean(Reward_pgd_attack_4))
        AdvStd_alpha.append(np.std(Reward_pgd_attack_4))
    Alpha_data = [CleanMean_alpha, CleanStd_alpha, AdvMean_alpha, AdvStd_alpha]
    RESULTS_LIST.append(Alpha_data)

save_file = open("results/TNNLS/CE_distilled_policy_evaluation_{}_tnnls.pkl".format(game),"wb")
pickle.dump(RESULTS_LIST,save_file)
save_file.close()