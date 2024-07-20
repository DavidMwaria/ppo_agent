# class used to wrap RL agent, replay memory

import Agent_PPO as PPO
from Agent_DQN import *
from PathCritic import *
from PatternCritic import *
from typing import Dict


class scheduler:
    def __init__(self, sch_type, mem_size, obs_dim, action_dim, lr, gamma, alpha, topo):
        self.type = sch_type
        self.rpm = RepMem(mem_size)
        if sch_type != 'path':
            self.serve_path_len = int(sch_type[1])
        else:
            self.serve_path_len = 0

        if sch_type == 'path':
            self.agent = PPO.Agent(0.99,0.2,0.95,0.0003,4,5,obs_dim,action_dim) # switched to PPO agent with built in actor and critic networks
            self.critic = self.agent.critic
        else:
            if   sch_type == '22':
                self.critic = PatternCritic22(obs_dim, action_dim)
            elif sch_type == '32':
                self.critic = PatternCritic32(obs_dim, action_dim)
            elif sch_type == '42':
                self.critic = PatternCritic42(obs_dim, action_dim)
            elif sch_type == '23':
                self.critic = PatternCritic23(obs_dim, action_dim)
            elif sch_type == '33':
                self.critic = PatternCritic33(obs_dim, action_dim)
            elif sch_type == '43':
                self.critic = PatternCritic43(obs_dim, action_dim)
            elif sch_type == '24':
                self.critic = PatternCritic24(obs_dim, action_dim)
            elif sch_type == '34':
                self.critic = PatternCritic34(obs_dim, action_dim)
            elif sch_type == '44':
                self.critic = PatternCritic44(obs_dim, action_dim)
            self.agent = Agent(
                critic=self.critic,
                obs_dim=obs_dim,
                action_dim=action_dim,
                lr=lr,
                gamma=gamma,
                alpha=alpha
            )
        self.sub_step = 0  # step counter for training
        if sch_type != 'path':
            self.ptn_dict = topo.pattern_dict[sch_type]

    # get pattern from dictionary by action index and config location to req
    def config_ptn(self, ptn_action, env):
        req = env.cur_req
        req.pattern_action = ptn_action  # record action result in req object
        #print("dict len= ", len(self.ptn_dict), "action =", ptn_action)
        pattern = self.ptn_dict[ptn_action]
        if ptn_action != 0:
            req.config_pattern(pattern)


class sch_list:
    def __init__(self, rpm_warmup_size, sch_dict: Dict[str, scheduler]):
        self.sch_dict: Dict[str, scheduler] = sch_dict
        self.all_warmup: bool = False
        self.rpm_warmup_size = rpm_warmup_size

    def check_warmup(self):
        result: bool = True
        for k in self.sch_dict.keys():
            if self.sch_dict[k].type=='path':          # PPO rpm would throw an error
                print('agent:', k,'(PPO)')
            else:
                print('agent:', k, 'rpm_size:', len(self.sch_dict[k].rpm))
        for k in self.sch_dict.keys():
            if (self.sch_dict[k].type!='path') and (len(self.sch_dict[k].rpm) < self.rpm_warmup_size): # no need to check rpm for path PPO
            #if len(self.sch_dict[k].rpm) < self.rpm_warmup_size and \
            #        self.sch_dict[k].serve_path_len == 3:
                result = False
                break
        self.all_warmup = True
        return result

    def step_reset(self):
        for k in self.sch_dict.keys():
            self.sch_dict[k].sub_step = 0


def make_sch_list(memory_warmup_size, mem_size, path_obs_dim, obs_dim, lr, gamma, alpha, topo):
    sch_dict: Dict[str, scheduler] = {}

    sch_dict['path'] = scheduler('path', mem_size, path_obs_dim, 6 + 1, lr, gamma, alpha, topo)
    sch_dict['22'] = scheduler('22', mem_size, obs_dim, 3 + 1, lr, gamma, alpha, topo)
    sch_dict['32'] = scheduler('32', mem_size, obs_dim, 4 + 1, lr, gamma, alpha, topo)
    sch_dict['42'] = scheduler('42', mem_size, obs_dim, 5 + 1, lr, gamma, alpha, topo)
    sch_dict['23'] = scheduler('23', mem_size, obs_dim, 6 + 1, lr, gamma, alpha, topo)
    sch_dict['33'] = scheduler('33', mem_size, obs_dim, 10 + 1, lr, gamma, alpha, topo)
    sch_dict['43'] = scheduler('43', mem_size, obs_dim, 15 + 1, lr, gamma, alpha, topo)
    sch_dict['24'] = scheduler('24', mem_size, obs_dim, 10 + 1, lr, gamma, alpha, topo)
    sch_dict['34'] = scheduler('34', mem_size, obs_dim, 20 + 1, lr, gamma, alpha, topo)
    sch_dict['44'] = scheduler('44', mem_size, obs_dim, 35 + 1, lr, gamma, alpha, topo)

    sch_lst = sch_list(memory_warmup_size, sch_dict)

    return sch_lst


# find pattern scheduler by 'm-n' string
def get_scheduler(env, path_action, sch_dict):
    env.cur_req.path_action = path_action
    ap_node = env.topo.ap[env.cur_req.src]
    pair_tuple = ap_node.get_pair_tuple(env.cur_req.dst)
    path = pair_tuple.pathSet[path_action]
    path_len = len(path) - 2  # exclude src and dst
    sfc_len = len(env.cur_req.vnf_seq)
    type_name = str(sfc_len) + str(path_len)
    target_sch = sch_dict[type_name]
    return target_sch
