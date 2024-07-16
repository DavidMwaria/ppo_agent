from typing import Dict, List
from EdgeTopo import *


# define virtual path class, it includes virtual links and computing nodes


class VirPath:
    def __init__(self, id, node_seq, nw_nodes, nw_links, llt):
        self.id = id  # local path id
        self.node_seq: List[int] = node_seq  # Sequence of nodes in this path
        self.nw_nodes = nw_nodes
        self.nw_links = nw_links
        self.llt = llt
        self.len = len(node_seq)  # Length of path, src & dst are not included
        self.avail_bw = 0  # available bandwidth limited by bottleneck link
        self.avail_cpu = 0  # available cpu number along this path
        self.delay = 0  # propagation of this path

        self.path_init()  # initialize members of the class
        # self.self_report()

    def path_init(self):
        self.bw_update()
        self.cpu_update()
        self.get_prop_delay()

    def self_report(self):
        print('path id: ', self.id)
        print('node sequence: ', self.node_seq)
        print('available BW: ', self.avail_bw)
        print('available CPU:', self.avail_cpu)
        print('propagation delay: ', self.delay)

    # get available bandwidth of this path updated
    def bw_update(self):
        self.avail_bw = self.nw_links[self.llt[self.node_seq[0]][self.node_seq[1]]].\
            get_BW_R(self.node_seq[0], self.node_seq[1])
        for i in range(len(self.node_seq) - 1):
            link = self.nw_links[self.llt[self.node_seq[i]][self.node_seq[i + 1]]]
            if link.get_BW_R(self.node_seq[i], self.node_seq[i+1]) < self.avail_bw:
                self.avail_bw = link.get_BW_R(self.node_seq[i], self.node_seq[i+1])
        return self.avail_bw

    # get available cpu numbers of nodes in this path updated
    def cpu_update(self):
        self.avail_cpu = 0
        for i in range(len(self.node_seq) - 2):
            self.avail_cpu += self.nw_nodes[self.node_seq[i + 1]].get_CPU_R()
        return self.avail_cpu

    # update possible patterns by marking corresponding bit to 0
    def pattern_update(self):
        pass

    # get propagation delay of traversing nodes and links of this path
    def get_prop_delay(self):
        # delay for traversing virtual links
        for i in range(len(self.node_seq) - 1):
            link = self.nw_links[self.llt[self.node_seq[i]][self.node_seq[i + 1]]]
            self.delay += link.get_latency()
        # delay for traversing nodes

        return self.delay
