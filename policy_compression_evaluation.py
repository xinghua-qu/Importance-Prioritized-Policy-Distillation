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
args.device = torch.device('cpu')

game_list   =  ['bank_heist', 'boxing', 'freeway', 'road_runner', 'pong','breakout','qbert']
evaluation_times = 30

hidden_dim_list = [256, 128, 64, 32, 16]
RESULTS = []
for game in game_list:
    args.game = game
    args.id = args.game
    args.evaluation_size = 1000
    args.seed = int(123)
    env = Env(args)
    env.eval()
    action_space = env.action_space()
    logfile = open('results/TNNLS/policy_compression_{}_kl_evaluation.txt'.format(game), 'a+')
    head = '%%%%%%%%%%%%%%% game = {}  policy compression %%%%%%%%%%%% \n'.format(game)
    logfile.write(head)
    game_evaluation_list = []
    for hidden_dim in hidden_dim_list:
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=(84,84,4)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(int(hidden_dim)))
        model.add(Activation('relu'))
        model.add(Dense(action_space))
        model.summary()   
        # load weights into model
        distiled_model_dir = "distiled_model_new/TNNLS/policy_compression_no_early_stop/{}/".format(game)
        save_name = "{}kl_hidden_dim_{}.hdf5".format(distiled_model_dir,hidden_dim)
        model.load_weights("{}".format(save_name))

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
        print('  Compression hidden dimension = {}\n'.format(hidden_dim))
        print('       {}\n'.format(Clean_reward))
        print('       No attack Reward_mean = {}; Reward_std = {}\n'.format(np.mean(Clean_reward), np.std(Clean_reward)))
        record = '  Compression hidden dimension = {}\n'.format(hidden_dim)
        logfile.write(record)
        record = '       {}\n'.format(Clean_reward)
        logfile.write(record)
        record = '       No attack Reward_mean = {}; Reward_std = {}\n'.format(np.mean(Clean_reward), np.std(Clean_reward))
        logfile.write(record)
        game_evaluation_list.append(Clean_reward)
    
    RESULTS.append(game_evaluation_list)
        
##### save the final results
save_file = open("results/TNNLS/policy_compression_kl_evaluation_tnnls.pkl","wb")
pickle.dump(RESULTS,save_file)
save_file.close()