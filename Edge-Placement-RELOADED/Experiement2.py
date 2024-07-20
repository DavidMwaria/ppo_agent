import sys
import math
import random
import copy

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# import self-defined modules
from EdgeTopo import *
from Env import *
from Baseline import *
from Agent_DQN import *
from Critic import Critic
from Scheduler import *
from Plot import *
from Validation import *
# Using CUDA 
use_cuda = torch.cuda.is_available() 
device   = torch.device("cuda" if use_cuda else "cpu")
print(device)
def train_model(sch_lst,memory_warmup_size, learn_freq, batch_size):
    update_num = 0
    
    # seprate call for PPO path agent
    for k in sch_lst.sch_dict.keys():
        if sch_lst.sch_dict[k].type!='path' and len(sch_lst.sch_dict[k].rpm) > memory_warmup_size and (sch_lst.sch_dict[k].sub_step % learn_freq == 0):
            (batch_obs, batch_action, batch_reward,
                 batch_next_obs, batch_done) = sch_lst.sch_dict[k].rpm.sample(batch_size)
            sch_lst.sch_dict[k].agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
            update_num+=1
            print("pattern learning...")
        elif sch_lst.sch_dict[k].type=='path' and sch_lst.sch_dict[k].sub_step % learn_freq == 0:
             sch_lst.sch_dict[k].agent.learn()
             print("PPO learning...")

    return update_num  # num of agent trained


def run_episode(env,env_HR,env_RH, sch_lst, memory_warmup_size, learn_freq, batch_size):
    total_reward = 0
    obs = env.reset()
    sch_lst.step_reset()
    pth_sch = sch_lst.sch_dict['path'] # get path scheduler
    ptn_sch = sch_lst.sch_dict['22'] # get pattern scheduler
    step = 0
    while True:
        step += 1
        '''path selection'''
        # sample path action from all possible actions
        if not env.use_path_h :
            if sch_lst.all_warmup:
                path_action, probs, value = pth_sch.agent.choose_action(obs)  # switched '.sample' to '.choose_action'
            else:
                path_action = h_path(env) # make rpm accumulating faster
            pth_sch.sub_step += 1
        else:
            path_action = h_path(env)
        # print('path_action = ', path_action)
        '''pattern selection'''
        if not env.use_pattern_h and path_action!=0 :
            ptn_sch = get_scheduler(env, path_action, sch_lst.sch_dict)
            pattern_action = ptn_sch.agent.sample(obs)
            if pattern_action != 0:
                ptn_sch.config_ptn(pattern_action, env)
            ptn_sch.sub_step += 1
        elif path_action!=0 :
            pattern_action =  h_pattern(env, path_action)
        else: # ignore pattern selection if path selection failed
            pattern_action = 0
        # deployment step
        next_obs, reward, done, _ = env.step(path_action)
        # append rpm
        if not env.use_path_h :      
            pth_sch.agent.store_data(obs,path_action,probs,value,reward,done)   # PPO memory
        if not env.use_pattern_h :
            ptn_sch.rpm.append((obs, pattern_action, reward, next_obs, done))

        # train model if needed
        train_model(sch_lst,memory_warmup_size, learn_freq, batch_size)
        # update reward and obs
        total_reward += reward
        obs = next_obs

        if done:
            if not plot_explicit:
                episode_rewards.append(total_reward)
                plot_rewards()
            break
    #env.stat.push_result() # write statistic data to file
    validate_episode(env,env_HR,env_RH, sch_lst, env.use_path_h, env.use_pattern_h) # validate using the same req list
    return total_reward


def train(continue_train=False,
          model_save_path='best_model', learn_freq=5, memory_size=20000,
          memory_warmup_size=2000, batch_size=32, learning_rate=0.001,
          gamma=0.4, alpha=0.9, max_episode=1000, ):

    topo1 = Topo(4, 10)
    env = Environment(topo1)
    env_HR = copy.deepcopy(env)  # env to simulate H path + RL pattern
    env_RH = copy.deepcopy(env)  # env to simulate RL path + H pattern
    env.stat.start_recording()
    # observation vector of network edges, nodes and current req
    obs_dim = 2*topo1.l_num + topo1.c_num + env.req_encode_size
    #print('obs_dim = ',obs_dim)
    sch_lst = make_sch_list(memory_warmup_size,memory_size, obs_dim, obs_dim, learning_rate, gamma, alpha,topo1)
    # pre-store some data in memory pool for warming up
    turn=0
    base_pth_h = True
    base_ptn_h = True
    env.base_reward = validate_episode(env,env_HR,env_RH, sch_lst,\
                                       base_pth_h, base_ptn_h)
    # warm up each agent by adding enough samples in its rpm
    if (not env.use_path_h) and (not env.use_pattern_h):
        while not sch_lst.check_warmup():
            print('turn = ', turn)
            run_episode(env,env_HR,env_RH, sch_lst, memory_warmup_size,
                        learn_freq, batch_size)
            turn+=1

    # start train
    print('training ...')
    episode = 0
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part
        for i in range(0, 100):
            total_reward = run_episode(
                env,env_HR,env_RH, sch_lst, memory_warmup_size, learn_freq,\
                batch_size)
            episode += 1

        print('total reward in episode ',episode,': ',total_reward)
    env.stat.end_recording()
    #agent.save(model_save_path)

print('Complete')
train(continue_train=False,
          model_save_path='best_model', learn_freq=5, memory_size=20000,
          memory_warmup_size=2000, batch_size=32, learning_rate=0.001,
          gamma=0.4, alpha=0.9, max_episode=600, )
if plot_explicit:
    plot_rewards(show_result=True)
    plt.ioff()
    plt.show()
