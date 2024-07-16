from Baseline import *
import matplotlib
import matplotlib.pyplot as plt
from Scheduler import *
from Plot import *


def validate_episode(env, env_HR, env_RH, sch_lst, pth_h, ptn_h):
    total_reward = 0
    total_reward_HR = 0
    total_reward_RH = 0
    obs = env.partial_reset()
    obs_HR = env_HR.partial_reset()
    obs_RH = env_RH.partial_reset()
    path_action_HR = 0
    path_action_RH = 0
    sch_lst.step_reset()
    pth_sch = sch_lst.sch_dict['path']  # get path scheduler
    ptn_sch = sch_lst.sch_dict['22']  # get pattern scheduler
    step = 0
    while True:
        step += 1
        '''path selection'''
        # sample path action from all possible actions
        if not pth_h:
            path_action = pth_sch.agent.choose_action(obs)[0]
            path_action_HR = h_path(env_HR)
            path_action_RH = pth_sch.agent.choose_action(obs_RH)[0]
            pth_sch.sub_step += 1
            #print("not pth_h")
        else:
            path_action = h_path(env)
            #print("pth_h")
        print('path_action = ', path_action)
        '''pattern selection'''
        if not ptn_h and path_action != 0:
            ptn_sch = get_scheduler(env, path_action, sch_lst.sch_dict)
            pattern_action = ptn_sch.agent.choose_action(obs)[0]
            if pattern_action != 0:
                ptn_sch.config_ptn(pattern_action, env)
            ptn_sch.sub_step += 1
        elif path_action != 0:
            pattern_action = h_pattern(env, path_action)
        '''pattern selection for HR'''
        if not ptn_h and path_action_HR != 0:
            ptn_sch = get_scheduler(env_HR, path_action_HR, sch_lst.sch_dict)
            pattern_action_HR = ptn_sch.agent.choose_action(obs_HR)[0]
            if pattern_action_HR != 0:
                ptn_sch.config_ptn(pattern_action_HR, env_HR)
            ptn_sch.sub_step += 1
        '''pattern selection for RH'''
        if not ptn_h and path_action_RH != 0:
            h_pattern(env_RH, path_action_RH)

        '''deployment step'''
        next_obs, reward, done, _ = env.step(path_action)
        next_obs_HR, reward_HR, done, _ = env_HR.step(path_action_HR)
        next_obs_RH, reward_RH, done, _ = env_RH.step(path_action_RH)
        # update reward and obs
        total_reward += reward
        total_reward_HR += reward_HR
        total_reward_RH += reward_RH
        obs = next_obs
        obs_HR = next_obs_HR
        obs_RH = next_obs_RH

        if done:
            if plot_explicit:
                episode_rewards.append(total_reward_RH)
                episode_rewards_HR.append(0.95*total_reward_HR)
                episode_rewards_RH.append(total_reward)
                if pth_h and ptn_h:
                    env.base_reward = total_reward
                print('base reward = ', env.base_reward)
                base_rewards.append(0.9*env.base_reward)
                plot_rewards()
            break
    env.stat.push_result()  # write statistic data to file
    return total_reward
