# implementation of environment class

from EdgeTopo import *
from ReqGen import *
from EpiStat import *

import gymnasium as gym
from gym import spaces
import numpy as np
import random
import math
import heapq
import copy

class Environment(gym.Env):
    def __init__(self, topology: Topo):
        self.cur_req = None  # request to be scheduled in each step
        self.topo = topology
        self.use_path_h: bool = False
        self.use_pattern_h: bool = False
        self.action_space = spaces.Discrete(4)  # each s-d pair has 3 paths for scheduling
        LOW_BOUND = np.zeros((51,), dtype=float)
        HIGH_BOUND = np.repeat(np.finfo(np.float32).max, 51)
        self.observation_space = spaces.Box(low=LOW_BOUND, high=HIGH_BOUND, \
                                            dtype=np.float32)
        self.cur_req_list = []
        self.total_req_list = self.topo.nw_req_list
        self.edge_num_factor = 0.9  # profit factor for edge num in a path
        self.image_dl_factor = 0.5  # profit discount factor for downloading image
        self.punish_flag = False
        # source one hot, sink one hot, bw, service time, total_cpu, path nodes one hot
        self.req_encode_size = self.topo.ap_num * 2 + 3 + self.topo.n_num
        # statistic entity
        self.stat = epi_stat()
        self.topo.history_reset()
        self.validate_req_list = self.topo.nw_req_list
        self.base_reward = 0  # horizontal baseline for comparison

    # reset network entities to initial state
    def reset(self):
        self.total_req_list = []
        for i in range(self.topo.ap_num):
            self.topo.ap[i].reset()
        for i in range(self.topo.c_num):
            self.topo.n[i + self.topo.ap_num].reset()
        for i in range(self.topo.l_num):
            self.topo.l[i].reset()
        self.topo.history_reset()
        self.total_req_list = self.topo.nw_req_list
        self.cur_req_list = []
        self.cur_req = heapq.heappop(self.total_req_list)
        self.stat.episode_reset()
        return self.make_observation()

    # reset network but keep the same req history
    def partial_reset(self):
        for i in range(self.topo.c_num):
            self.topo.n[i + self.topo.ap_num].reset()
        for i in range(self.topo.l_num):
            self.topo.l[i].reset()
        self.total_req_list = copy.deepcopy(self.validate_req_list)
        self.cur_req_list = []
        self.cur_req = heapq.heappop(self.total_req_list)
        self.stat.episode_reset()
        return self.make_observation()

    # deploy SFC req to selected path                   
    def sfc_deploy(self, req: SFC_req):
        path_action = req.path_action
        ap_node = self.topo.ap[req.src]
        pair_tuple = ap_node.get_pair_tuple(req.dst)
        path = pair_tuple.pathSet[path_action]  # path to be deployed
        # pattern location finished in req, no operation here
        # print('deploy req: ', req.vnf_id,' along ', path)
        # print('pattern: ', pattern
        for i in range(req.sfc_len):  # deploy VNF container in each C_node
            # print('i=', i, 'node = ',path[req.vnf_seq[i].location])
            self.topo.n[path[req.vnf_seq[i].location]].VNF_alloc(req, i)
            # print('i=', i)
        for i in range(len(path) - 1):  # deploy bw to edges along the path
            link = self.topo.l[self.topo.llt[path[i]][path[i + 1]]]
            link.config_traffic(path[i], path[i + 1], req.bw)
            # update residual bw of each path after deployment

    def req_remove(self, req: SFC_req):
        # print('remove req =', req.vnf_id)
        action = req.path_action
        ap_node = self.topo.ap[req.src]
        pair_tuple = ap_node.get_pair_tuple(req.dst)
        action_path = pair_tuple.pathSet[action]  # path to be effected
        # print('affected path=', action_path)
        for i in range(len(action_path) - 2):  # delete container in each C_node and free CPU
            # print('In node ',action_path[i+1],':')
            self.topo.n[action_path[i + 1]].VNF_clear(req.vnf_id)
            # print('i=', i)
        for i in range(len(action_path) - 1):  # resume bw to edges along the path
            link = self.topo.l[self.topo.llt[action_path[i]][action_path[i + 1]]]
            link.config_traffic(action_path[i], action_path[i + 1], -req.bw)

    # remove finished reqs before next arrival scheduling
    def update_env(self, next_arr_TS):
        cur_list = self.cur_req_list
        pop_num = 0  # number of reqs popped, also used as offset after del
        for i in range(len(cur_list)):
            # print('i=', i)
            if cur_list[i - pop_num].leave_TS < next_arr_TS:
                # print(cur_list[i-pop_num].vnf_id)
                self.req_remove(cur_list[i - pop_num])
                del cur_list[i - pop_num]
                pop_num += 1
                if (i - pop_num) >= len(cur_list):
                    break
        return pop_num

    def reward(self, req: SFC_req):
        action = req.path_action
        ap_node = self.topo.ap[req.src]
        pair_tuple = ap_node.get_pair_tuple(req.dst)
        if not action == 0:  # accept if action is not 0
            action_path = pair_tuple.pathSet[action]  # path chosen for this req

            extra_cpu_factor = req.extra_cpu_factor()
            if not req.image_deferred:
                edge_num = len(action_path) - 1  # count edge num in path
                reward = req.bw * req.duration * \
                         math.pow(self.edge_num_factor, edge_num) * \
                         extra_cpu_factor
                return reward
            else:
                edge_num = len(action_path) - 1  # count edge num in path
                reward = req.bw * req.duration * \
                         math.pow(self.edge_num_factor, edge_num) * \
                         extra_cpu_factor * self.image_dl_factor
                return reward
        else:
            reward = req.bw * req.duration  # base reward of req for penalty
            return reward

    def req_2_vector(self, req: SFC_req):
        cur_req_encode = np.zeros(self.req_encode_size, dtype=np.float32)
        source = req.src
        sink = req.dst
        cur_req_encode[source] = 1
        cur_req_encode[self.topo.ap_num + sink] = 1
        cur_req_encode[self.topo.ap_num * 2] = req.bw
        cur_req_encode[self.topo.ap_num * 2 + 1] = req.duration
        cur_req_encode[self.topo.ap_num * 2 + 2] = req.cpu_total
        # print('path:', req.path_action, 'pattern:', req.pattern_action)
        if req.path_action != 0 and req.pattern_action != 0:
            ap_node = self.topo.ap[req.src]
            pair_tuple = ap_node.get_pair_tuple(req.dst)
            path = pair_tuple.pathSet[req.path_action]  # path to be deployed
            for i in range(len(path)):
                cur_req_encode[self.topo.ap_num * 2 + 3 + path[i]] = 1

        return cur_req_encode

    def transform_list(self, req_list: List[SFC_req]):
        random.shuffle(req_list)
        req_vector_list = []
        for req in req_list:
            req_vector_list.append(self.req_2_vector(req))
        return np.expand_dims(np.vstack(req_vector_list), axis=0)

    def make_observation(self):
        # edge state: residual bandwidth of links in both directions
        remain_bd_list = []
        for i in range(self.topo.l_num):
            x, y = self.topo.l[i].get_endpoint()
            remain_bd_list.append(self.topo.l[i].get_BW_R(x, y))
            remain_bd_list.append(self.topo.l[i].get_BW_R(y, x))
        edge_state_encode = np.array(remain_bd_list, dtype=np.float32)

        # computing node state: residual CPU cores
        remain_cpu_list = []
        offset = self.topo.ap_num  # AP node excluded by offset
        for i in range(self.topo.c_num):
            r_cpu = self.topo.n[i + offset].get_CPU_R()
            remain_cpu_list.append(r_cpu)
        cpu_state_encode = np.array(remain_cpu_list, dtype=np.float32)

        # current request waiting for scheduling
        cur_req_encode = np.array(self.req_2_vector(self.cur_req), dtype=np.float32)

        obs_nw = np.concatenate([edge_state_encode, cpu_state_encode])
        obs_req = cur_req_encode
        obs_total = np.concatenate([obs_nw, obs_req])
        # obs_batch = torch.from_numpy(obs_total.astype(np.float32)).view(1,-1)

        return obs_total

    # check whether bandwidth and cpu resources are sufficient on selected path,
    # If not, then reject this request.
    def feasible_check(self, path_action, pattern_action):
        actual_action = 0
        #print("feasible_check: ", path_action)
        if path_action == 0 or pattern_action == 0:
            return actual_action
        req = self.cur_req
        ap_node = self.topo.ap[req.src]
        pair_tuple = ap_node.get_pair_tuple(req.dst)
        path = pair_tuple.pathSet[path_action]  # path to be deployed
        path_obj = pair_tuple.path_list[path_action]
        # bandwidth check
        for i in range(len(path) - 1):  # deploy bw to edges along the path
            link = self.topo.l[self.topo.llt[path[i]][path[i + 1]]]
            if link.get_BW_R(path[i], path[i + 1]) < req.bw:
                #print('bw overloaded!\n')
                return actual_action
        # residual CPU num check
        path_cpu_r = [0]  # append src=0 for consistency only
        for i in range(len(path) - 2):  # get cpu_r of nodes in this path
            path_cpu_r.append(self.topo.n[path[i + 1]].CPU_R)
        for i in range(req.sfc_len):
            path_cpu_r[req.vnf_seq[i].location] -= req.vnf_seq[i].cpu_total
            if path_cpu_r[req.vnf_seq[i].location] < 0:
                #print('cpu in node not enough!\n')
                return actual_action

        actual_action = 1  # all checks passed, allow for deployment
        return actual_action

    def step(self, path_action):
        finished = False
        self.cur_req.path_action = path_action
        pattern_action = self.cur_req.pattern_action
        # print('req: ', self.cur_req.vnf_id,'act: ',self.cur_req.action)
        # check whether selected action path is feasible
        dp_action = self.feasible_check(path_action, pattern_action)
        # accept        
        reward = self.reward(self.cur_req)
        if dp_action != 0:
            self.sfc_deploy(self.cur_req)  # deploy along action path
            heapq.heappush(self.cur_req_list, self.cur_req)
        else:
            if self.punish_flag:
                reward *= -1.
            else:
                reward = 0
        self.stat.add_record(dp_action, reward)
        # finished
        if len(self.total_req_list) <= 1:
            finished = True
            return self.make_observation(), reward, finished, {}
        # update current request
        self.cur_req = heapq.heappop(self.total_req_list)
        # updated network state, pop out all leaved requests before next arrival TS
        next_arr_TS = self.cur_req.arr_TS
        if len(self.cur_req_list) > 0:
            pop_num = self.update_env(next_arr_TS)
            # print(pop_num, " requests left network before next arrival")

        return self.make_observation(), reward, finished, {}
