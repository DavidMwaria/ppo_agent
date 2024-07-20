# Generate SFC request list in time sequence
import random
import numpy as np
from typing import Dict, List
from SDLib import *
from VirtualPath import *


class VNF_req:
    def __init__(self, image: str):
        self.location: int = -1  # node id for deployment
        self.image: str = image
        self.replica_enable: bool = False
        self.boost_enable: bool = False
        self.cpu_basic = 2
        self.cpu_boost = 0  # extra cpu cores for processing boost
        self.cpu_replica = 0  # extra cpu cores for replication
        self.attr_init()  # config replica and boost options
        self.cpu_primary = self.cpu_basic + self.cpu_boost  # actual cpu usage of this vnf
        self.cpu_total = self.cpu_primary + self.cpu_replica

    def attr_init(self):  # whether this VNF support replica and performance boost
        if self.image[1] == '1':
            self.replica_enable = True
            self.cpu_replica = 1
        if self.image[2] == '1':
            self.boost_enable = True

    def boost_config(self, boost_num):
        if self.boost_enable:
            self.cpu_boost = boost_num
            self.cpu_primary = self.cpu_basic + self.cpu_boost
            self.cpu_total = self.cpu_primary + self.cpu_replica


# define Service Function Chain request
class SFC_req:
    def __init__(self, req_id, src, dst, bw, avail, arr_TS, leave_TS):
        self.req_id = req_id  # local id of this request
        self.src = src
        self.dst = dst
        self.bw = bw  # minimal bandwidth requirement
        self.Avail = avail  # minimal availability requirement
        self.arr_TS = arr_TS  # arrival timestamp
        self.leave_TS = leave_TS  # leave timestamp
        self.cpu_total = 0  # sum of cpu cores of all VNFs
        # data derived from base description
        self.vnf_id = str(self.src) + '-' + str(self.dst) + '-' + str(self.req_id)  # global id identified by C_node
        self.duration = leave_TS - arr_TS  # duration of serving times
        self.sfc_len = 0  # the number of VNFs in this SFC, range: 2 to 4
        self.state = 0  # serving state code, 0: unscheduled, 1: scheduled, 2: rejected.
        self.delay_bound = 10  # maximal delay bound
        self.vnf_seq = []  # sequence of VNFs
        self.image_seq = []  # sequence of image of each VNF
        self.cpu_req_basic = []  # minimum cpu req for each VNF
        self.cpu_req_actual = []  # actual cpu usage for each VNF
        self.seq_init()  # create vnf sequence randomly
        self.dp_pattern = []  # final deploy pattern
        self.image_deferred = False  # set to True if C_node does not have image in memory
        self.fun_len_mask: int  # function length mask
        # deployment plan of this req
        self.acceptance = False
        self.path_action = 0  # action path number
        self.pattern_action = 0  # pattern number
        # self report for debugging
        self.report_enable = False
        if self.report_enable:
            self.self_report()

    def __lt__(self, other):  # compare requests by their arrival time
        return self.arr_TS < other.arr_TS

    def seq_init(self):  # initialize sequence of VNFs of this SFC
        new_seq = []
        self.sfc_len = random.randint(2, 4)  # the number of VNFs in this SFC, range: 2 to 4
        for i in range(self.sfc_len):
            image_dict = vnf_dict[i]
            vnf = VNF_req(image_dict[random.randint(1, 4)])
            self.cpu_req_basic.append(vnf.cpu_basic)
            self.cpu_req_actual.append(vnf.cpu_total)
            self.cpu_total += vnf.cpu_total
            self.image_seq.append(vnf.image)
            new_seq.append(vnf)
        self.vnf_seq = new_seq

    def config_replica(self, replica_list):  # assign replicas to each VNF for availability
        for i in range(len(self.vnf_seq)):
            self.vnf_seq[i] += replica_list[i]  # add one replica to each VNF

    def config_pattern(self, pattern: str):  # config VNF containers along the path following the pattern
        self.dp_pattern = pattern
        offset = 0
        # print('pattern: ', pattern, 'sfc_len:', self.sfc_len)
        ncn = pattern[offset]  # container Num of Current Node
        for i in range(self.sfc_len):
            # print('offset:', offset, 'ncn:', ncn)
            while ncn == 0:
                offset += 1
                ncn = pattern[offset]
            if ncn > 0:
                self.vnf_seq[i].location = offset + 1  # deployment start after src node
                ncn -= 1
            while ncn == 0:
                offset += 1
                if offset == len(pattern):
                    break
                ncn = pattern[offset]

    def extra_cpu_factor(self):  # return discount factor for extra CPU consumption
        factor = 1
        for i in range(self.sfc_len):
            factor *= self.cpu_req_basic[i] / self.cpu_req_actual[i]
        return factor

    def self_report(self):  # report available info of this request
        print('req id: ', self.vnf_id)
        print('vnf sequence: ', self.image_seq)
        print('BW req:', self.bw)
        print('CPU req:', self.cpu_total)
        print('admission result: ', self.acceptance)
        # give deployment plan if req is accepted
        if self.acceptance:
            print('deploy path: ', self.path_action)
            print('cpu config: ', self.cpu_total)
            print('deploy pattern: ', self.dp_pattern)
            print('operation state: ', self.state)


# manage sfc_req list and path info
class ReqGen_tuple:
    def __init__(self, total_time, AP_src, AP_dst, p_lambda, e_mu,
                 path_set_dict, node_list, link_list, llt):
        self.pathSet = None
        self.path_list = None
        self.totalTime = total_time  # total time for simulation
        self.src = AP_src
        self.dst = AP_dst
        self.pair_id = str(self.src) + '-' + str(self.dst)
        self.prb: Dict[int, float]  # residual bandwidth of each path
        self.p_lambda = p_lambda  # parameters of poisson process
        self.e_mu = e_mu  # parameters of exponential distribution
        self.arr_record = np.random.exponential(3, 200)  # create arrival interval record
        self.serve_record = np.random.exponential(10, 200)  # create serve time record
        self.bw_record = np.random.exponential(10, 200)  # create bandwidth record
        self.it_a = iter(self.arr_record)
        self.it_b = iter(self.bw_record)
        self.it_d = iter(self.serve_record)
        self.avail = [0.95, 0.99, 0.999, 0.9995]  # choose avail for a req randomly
        self.node_list = node_list
        self.link_list = link_list
        self.llt = llt  # link lookup table
        self.add_path_set(path_set_dict)  # add path set for this s-d pair from path set dict

        self.arrList = []  # history list of arrival time point of req
        self.serveList = []  # current request in serving
        self.make_history()
        # self.print_history()
        self.pathLenMask: Dict[int, int]  # mask of path length

    # add candidate paths for scheduling 
    def add_path_set(self, path_set_dict):
        # pathSet: node seq only, path_list: set of virPath obj
        self.pathSet: Dict[int, List[str]] = path_set_dict[self.pair_id]
        self.path_list: Dict[int, VirPath] = {}
        for i in range(len(self.pathSet)):
            self.path_list[i + 1] = VirPath(i + 1, self.pathSet[i + 1],
                                            self.node_list, self.link_list, self.llt)
        # print("Path set of pair ", self.pair_id,": ")
        # for i in range(len(self.pathSet)):
        #    print(self.pathSet[i+1]) #print dict key from 1 to 3

    def reset(self):
        self.arrList.clear()
        self.arrList = self.make_history()

    # make arrival record during the simulation
    def make_history(self):
        arr_ts = next(self.it_a)
        duration = round(next(self.it_d)) + 1
        leave_ts = arr_ts + duration
        req_id = 0
        while leave_ts < self.totalTime:
            request = SFC_req(req_id, self.src, self.dst,
                              round(1 + next(self.it_b)), self.avail[0], arr_ts, leave_ts)
            self.arrList.append(request)
            arr_ts = arr_ts + next(self.it_a)
            duration = round(next(self.it_d)) + 1
            leave_ts = arr_ts + duration
            req_id += 1

        return len(self.arrList)

    def print_history(self):
        filename="req_history_"+self.pair_id+".txt"
        fd = open(filename, "w+")
        #print("ID  src   dst   arr   leave   bw   str")
        str1 = "ID  src   dst   arr   leave   bw   str\n"
        fd.write(str1)
        for i in range(len(self.arrList)):
            r = self.arrList[i]
            #print(r.req_id, " ", r.src, "   ", r.dst,
            #      "   ", r.arr_TS, "   ", r.leave_TS, "   ", r.bw, "   ", r.vnf_id)
            str2 =str(r.req_id)+" "+str(r.src)+"   "+str(r.dst)+\
                 "   "+str(r.arr_TS)+"   "+ str(r.leave_TS)\
                  +"   "+ str(r.bw)+ "   "+ str(r.vnf_id)+'\n'
            fd.write(str2)
        #print("request num = ", len(self.arrList))
        str3="total request num = "+ str(len(self.arrList))+'\n'
        fd.close()

# request generator produce requests in time sequence
