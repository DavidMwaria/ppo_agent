# Baseline scheduling algorithm: a heuristic version
from EdgeTopo import *
from Env import *
from SDLib import *


# heuristic path selection
def h_path(env: Environment):
    cur_req = env.cur_req
    ap_node = env.topo.ap[cur_req.src]
    pair_tuple = ap_node.get_pair_tuple(cur_req.dst)
    path_list = pair_tuple.path_list
    path_action = 0
    max_r_bw = 0  # find path with maximal residual bw
    for i in range(len(path_list)):
        if path_list[i + 1].avail_bw > max_r_bw and path_list[i + 1].avail_cpu > cur_req.cpu_total:
            max_r_bw = path_list[i + 1].avail_bw
            path_action = i + 1

    return path_action


# heuristic pattern selection: complexity: O(mn)
def h_pattern(env: Environment, path_action):
    pattern_action = 1
    req = env.cur_req
    vnf_seq = req.vnf_seq
    req.path_action = path_action
    ap_node = env.topo.ap[req.src]
    node = env.topo.n
    pair_tuple = ap_node.get_pair_tuple(req.dst)
    # print('path action = ', path_action)
    node_seq = pair_tuple.pathSet[path_action]  # get path to be deployed
    pattern = []
    tmp_node_cpu_r = []  # list for storing residual CPU_R of each node
    for i in range(len(node_seq)-2):
        pattern.append(0)
        tmp_node_cpu_r.append(node[node_seq[i+1]].CPU_R)
    offset = 0
    found = False
    if len(tmp_node_cpu_r) != (len(node_seq) - 2):
        print('len1=',tmp_node_cpu_r,'len2=',node_seq)
    for i in range(len(vnf_seq)):
        found = False
        while offset < (len(node_seq)-2):
            if tmp_node_cpu_r[offset] >= vnf_seq[i].cpu_total:
                tmp_node_cpu_r[offset] -= vnf_seq[i].cpu_total
                pattern[offset] += 1
                found = True
                break
            else:
                offset += 1  # move to next node if current node is full

        if not found:
            pattern_action = 0  # deploy plan not found
            break
    req.pattern_action = pattern_action  # record action result, even not used in heuristic version
    if found:
        req.config_pattern(pattern)

    return pattern_action


# heuristic pattern selection
def h_pattern_v2(env: Environment, path_action):
    pattern_action = 1
    req = env.cur_req
    vnf_seq = req.vnf_seq
    req.path_action = path_action
    ap_node = env.topo.ap[req.src]
    pair_tuple = ap_node.get_pair_tuple(req.dst)
    print('path action = ', path_action)
    path = pair_tuple.pathSet[path_action]  # get path to be deployed
    key = str(req.sfc_len) + '-' + str(len(path) - 2)
    pattern_set = m_n_pattern_dict[key]
    print('candidate patterns: ', pattern_set)

    # pattern_mask = feasible_pattern()
    # pattern_action = min()
    pattern = pattern_set[pattern_action]
    # config pattern to make vnf location
    req.config_pattern(pattern)

    return pattern_action
