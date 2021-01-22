import torch
import argparse
from env import Env
import atari_py

from agent import Agent
import os
import numpy as np
import pickle
import h5py

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
# parser.add_argument('--lb', type=int, default=20,  help='lower bound of reward for collect data')

def store_many_hdf5(states, qvalues, game):
    """ Stores an array of states to HDF5.
        Parameters:
        ---------------
        states       images array, (N, 84, 84, 4) to be stored
        qvalues      qvalues array, (N, acion_dimension) to be stored
    """
    num_images = len(states)

    # Create a new HDF5 file
    file = h5py.File("atari_data/{}.h5".format(game), "w")

    # Create a dataset in the file
    dataset = file.create_dataset("states", np.shape(states), h5py.h5t.STD_U8BE, data=states)
    meta_set = file.create_dataset("qvalues", np.shape(qvalues), h5py.h5t.STD_U8BE, data=qvalues)
    file.close()

# Setup
args = parser.parse_args()
args.evaluation_episodes = 100
args.model = 'teacher_policies/{}.pth'.format(args.game)
args.memory = 'teacher_policies/{}.pth'.format(args.game)
args.device = torch.device('cuda')
args.hidden_size = 512
args.id = args.game
args.evaluation_size = 1000
args.seed = int(123)

env = Env(args)
env.eval()

action_space = env.action_space()

# Agent
dqn = Agent(args, env)
dqn.eval()

STATES_buffer  = []
ACTIONS_buffer = []
REWARD_buffer = []
data_lenth = int(1e5)
game = args.game
logfile = open('atari_data/{}_data_collection_reward.txt'.format(game), 'a+')
head = '%%%%%%%%%%%%%%% game = {} %%%%%%%%%%%% \n'.format(game)
logfile.write(head)
# Test performance over several episodes
done = True
for j in range(1000):
    Episode_state = []
    Episode_action = []
    while True:
        if done:
            state, reward_sum, done = env.reset(), 0, False
        list_state_1 = state.tolist()
        list_state = np.array(list_state_1)
        list_state = list_state*255
        list_state = list_state.astype(int)
        Episode_state.append(list_state)
        qvalues = (dqn.online_net(state.unsqueeze(0)) * dqn.support).sum(2)
        qvalues = qvalues.tolist()
        Episode_action.append(qvalues[0])
        action = np.argmax(qvalues[0])
        state, reward, done = env.step(action)  # Step
        reward_sum += reward

        if done:            
            REWARD_buffer.append(reward_sum)
            message = "         Episode {} reward:   {} \n".format(j, reward_sum)
            logfile.write(message)
            print('Total reward on {}: {}'.format(args.game,reward_sum))
            break
    STATES_buffer  = STATES_buffer + Episode_state
    ACTIONS_buffer = ACTIONS_buffer + Episode_action  
    if len(STATES_buffer)>=data_lenth:
        ######## data saving path #########
        data_dir = '/home/teddy/policy_distillation/policy_distillation/atari_data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        STATES_buffer  = np.array(STATES_buffer)
        ACTIONS_buffer = np.array(ACTIONS_buffer)
        store_many_hdf5(STATES_buffer, ACTIONS_buffer, args.game)
        REWARD_buffer = np.array(REWARD_buffer)
        message = "Reward mean on {}:   {} \n".format(game, np.mean(REWARD_buffer))
        logfile.write(message)
        message = "Stard deviation on {}:   {} \n".format(game, np.std(REWARD_buffer))
        logfile.write(message)
        break
    env.close()