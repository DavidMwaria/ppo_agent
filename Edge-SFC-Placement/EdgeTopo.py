import numpy as np
import heapq
from typing import Dict, List
# import self-defined entities
from EdgeEnt import *  # AP_node,C_node,D_link
from SDLib import m_n_pattern_dict


class Topo:
    def __init__(self, ap_num, c_num):
        self.path_set_dict = None
        self.ap_num = ap_num  # number of edge access nodes
        self.c_num = c_num  # number of computing nodes
        self.n_num = ap_num + c_num  # number of nodes in the topology 
        # link lookup table used to return link id given two endpoint
        self.llt = np.full((self.n_num, self.n_num), -1, dtype=int)
        self.ap = []  # list of AP nodes
        self.n = []  # list of computing nodes
        self.l = []  # list of logical links
        self.link_obs = []  # vector state of residual graph of network link
        self.node_obs = []  # vector state of residual capacity of computing node
        # 'm-n': patterns of m functions in n-hop path
        self.pattern_dict = m_n_pattern_dict
        self.topo_init()
        self.destNum = 2
        # make req history and combine into one list
        for i in range(ap_num):
            if i < (ap_num / 2):
                self.ap.append(AP_node(i, 'src'))
                #self.ap[i].ReqPair_init(self.destNum, 3, 3, self.path_set_dict,
                #                       self.n, self.l, self.llt)
                self.n.append(AP_node(i, 'src'))  # for index consistency, not used
            else:
                self.ap.append(AP_node(i, 'dst'))
                self.n.append(AP_node(i, 'dst'))  # for index consistency, not used
        self.nw_req_gen()
        # self.nw_req_print()
        for i in range(c_num):
            self.n.append(C_node(ap_num + i, 5))
            # self.n[i].self_report()
        # initialize ReqGen tuples in src APs
        for i in range(ap_num):
            if i < (ap_num / 2):
                self.ap[i].ReqPair_init(self.destNum, 3, 3, self.path_set_dict,
                                        self.n, self.l, self.llt)

    def history_reset(self):
        self.destNum = 2
        # remake req history and combine into one list
        for i in range(self.ap_num):
            if i < (self.ap_num / 2):
                self.ap[i].ReqPair_init(self.destNum, 3, 3, self.path_set_dict,
                                        self.n, self.l, self.llt)
        self.nw_req_gen()

    def nw_req_gen(self):  # merge req_tuples of all src APs
        self.nw_req_list = []
        # print("src ap num=",int(self.ap_num/2))
        src_ap_num = int(self.ap_num / 2)
        for i in range(src_ap_num):
            for j in range(len(self.ap[i].pairList)):
                for k in range(len(self.ap[i].pairList[j].arrList)):
                    heapq.heappush(self.nw_req_list, self.ap[i].pairList[j].arrList[k])

    def nw_req_print(self):
        print("total request num = ", len(self.nw_req_list))
        print("ID  src   dst   arr                   leave \
              bw     str")
        for i in range(len(self.nw_req_list)):
            r = heapq.heappop(self.nw_req_list)
            print(r.req_id, " ", r.src, "   ", r.dst,
                  "   ", r.arr_TS, "   ", r.leave_TS, "   ", r.bw, "   ", r.vnf_id)

    def print_topo(self):  # return topology in matrix
        return print(self.link_map)

    # fill link lookup table
    def fill_llt(self):
        for i in range(len(self.l)):
            self.llt[self.l[i].A_id][self.l[i].B_id] = self.l[i].link_id
            self.llt[self.l[i].B_id][self.l[i].A_id] = self.l[i].link_id
        return 0  # print(self.llt)

    def get_R_CPU(self):  # get residual CPU cores of all computing nodes
        self.node_obs = []
        for i in range(self.c_num):
            self.node_obs.append(self.n[i + self.ap_num].get_CPU_R())
        return self.node_obs

    def get_R_BW(self):  # get residual bandwidth of all links
        self.link_obs = []
        for i in range(self.l_num):
            # print("link ",i,":",self.l[i].get_BW_R())
            self.link_obs.append(self.l[i].get_BW_R())
        return self.link_obs

    def topo_init(self):
        """ Matrix "link_map" denotes the link id of two adjacent node,
        the value will be -1 if no connection exists. """
        self.link_map = (-1) * np.ones((self.n_num, self.n_num))
        # initiating topology connections        
        self.l.append(D_link(0, 0, 4, self.link_map))
        self.l.append(D_link(1, 0, 5, self.link_map))
        self.l.append(D_link(2, 1, 5, self.link_map))
        self.l.append(D_link(3, 1, 6, self.link_map))
        self.l.append(D_link(4, 11, 2, self.link_map))
        self.l.append(D_link(5, 12, 2, self.link_map))
        self.l.append(D_link(6, 12, 3, self.link_map))
        self.l.append(D_link(7, 13, 3, self.link_map))

        self.l.append(D_link(8, 4, 7, self.link_map))
        self.l.append(D_link(9, 4, 8, self.link_map))
        self.l.append(D_link(10, 4, 9, self.link_map))
        self.l.append(D_link(11, 5, 7, self.link_map))
        self.l.append(D_link(12, 5, 8, self.link_map))
        self.l.append(D_link(13, 5, 9, self.link_map))
        self.l.append(D_link(14, 5, 10, self.link_map))
        self.l.append(D_link(15, 6, 8, self.link_map))
        self.l.append(D_link(16, 6, 9, self.link_map))
        self.l.append(D_link(17, 6, 10, self.link_map))

        self.l.append(D_link(18, 7, 11, self.link_map))
        self.l.append(D_link(19, 8, 11, self.link_map))
        self.l.append(D_link(20, 9, 11, self.link_map))
        self.l.append(D_link(21, 7, 12, self.link_map))
        self.l.append(D_link(22, 8, 12, self.link_map))
        self.l.append(D_link(23, 9, 12, self.link_map))
        self.l.append(D_link(24, 10, 12, self.link_map))
        self.l.append(D_link(25, 8, 13, self.link_map))
        self.l.append(D_link(26, 9, 13, self.link_map))
        self.l.append(D_link(27, 10, 13, self.link_map))

        self.l.append(D_link(28, 4, 11, self.link_map))
        self.l.append(D_link(29, 5, 12, self.link_map))
        self.l.append(D_link(30, 6, 13, self.link_map))

        self.l.append(D_link(31, 8, 7, self.link_map))
        self.l.append(D_link(32, 8, 9, self.link_map))
        self.l.append(D_link(33, 9, 10, self.link_map))

        self.l_num = len(self.l)  # number of logical links
        self.fill_llt()  # create link lookup table

        self.path_set_dict: Dict[str, Dict[int, List[int]]] = \
            {'0-2': {1: [0, 4, 11, 2],
                     2: [0, 5, 12, 2],
                     3: [0, 4, 7, 11, 2],
                     4: [0, 5, 8, 11, 2],
                     5: [0, 5, 8, 12, 2],                     
                     6: [0, 4, 8, 7, 11, 2], },
             '0-3': {1: [0, 5, 12, 3], 
                     2: [0, 4, 8, 12, 3],
                     3: [0, 5, 9, 12, 3],
                     4: [0, 5, 9, 13, 3],
                     5: [0, 4, 8, 9, 12, 3],
                     6: [0, 5, 9, 10, 13, 3],},
             '1-2': {1: [1, 5, 12, 2],
                     2: [1, 5, 8, 11, 2],
                     3: [1, 5, 8, 12, 2],
                     4: [1, 6, 9, 12, 2],
                     5: [1, 6, 9, 8, 11, 2],
                     6: [1, 5, 8, 7, 12, 2], },
             '1-3': {1: [1, 6, 13, 3],
                     2: [1, 5, 12, 3],
                     3: [1, 6, 10, 13, 3],
                     4: [1, 5, 9, 13, 3],
                     5: [1, 5, 9, 12, 3],
                     6: [1, 6, 9, 10, 13, 3], }
             }

