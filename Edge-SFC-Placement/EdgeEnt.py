# define entities of edge computing network

import math
import random
from ReqGen import SFC_req, ReqGen_tuple


# define access point class
class AP_node:
    def __init__(self, node_id, service_type):
        self.pairList = []
        self.node_id = node_id
        self.service_type = service_type
        self.pair_num = 0

    def self_report(self):
        print("I am AP node %d, my service type is %s. " % \
              (self.node_id, self.service_type))
        print('pair num =', self.pair_num)
        for i in range(len(self.pairList)):
            print(self.pairList[i].pair_id)

    def reset(self):
        self.pairList.clear()
        self.pair_num = 0

    # initiate src-dst pair for request generation
    def ReqPair_init(self, dest_num, p_lambda, e_mu,
                     path_set_dict, node_list, link_list, llt):
        self.pair_num = 0
        for i in range(dest_num):
            new_pair = ReqGen_tuple(200, self.node_id, self.pair_num + dest_num,
                                    p_lambda, e_mu, path_set_dict, node_list, link_list, llt)
            self.pairList.append(new_pair)
            self.pair_num += 1
            # self.pairList[i].print_path_set()
        # print("s-d pair of AP node",self.node_id,":",pair_num)
        return self.pair_num

    def get_pair_tuple(self, dst):  # get tuple by dst
        for i in range(len(self.pairList)):
            if self.pairList[i].dst == dst:
                return self.pairList[i]


# define container class for VNF
class VNF_C:
    def __init__(self, vnf_id, cpu_assign, image):
        self.vnf_id = vnf_id  # VNF id denoted by string
        self.VNF_type = 1  # the type of VNF
        self.cpu_num = cpu_assign  # how many cpus assigned to this VNF
        self.P_delay = 0  # processing delay of traffic passing this VNF
        self.image = image
        self.running_state = 'off'

    def deploy(self):  # install container with its replicas
        self.running_state = 'on'

    def get_delay(self, traffic):  # get processing delay
        self.P_delay = traffic / (self.cpu_num * self.VNF_type)
        return self.P_delay


# define computing node class
class C_node:
    def __init__(self, node_id, container_num):
        self.CPU_num = 32  # cpu cores owned by this node
        self.CPU_R = 32  # residual cpu cores available
        self.node_id = node_id
        self.container_num = container_num
        self.MEM_size = 256  # memory in GB installed in this node
        self.MEM_list = []  # list of VNF images available in this node,size fixed
        self.VNF_list = []

    def reset(self):
        self.CPU_R = self.CPU_num
        self.MEM_list.clear()
        self.VNF_list.clear()

    def self_report(self):
        print("I am container node %d, I can hold %d containers. "
              % (self.node_id, self.container_num))

    def state_report(self):
        print("The container usage of node %d is: " % self.node_id)

    def VNF_alloc(self, req: SFC_req, index):
        container = VNF_C(req.vnf_id, req.vnf_seq[index].cpu_total, req.vnf_seq[index].image)
        container.deploy()  # install the container with its replicas
        self.CPU_R = self.CPU_R - container.cpu_num
        self.VNF_list.append(container)

    def VNF_clear(self, del_id):
        result = 0
        # print('** del vnf id=', del_id)
        # for i in range(len(self.VNF_list)):
        #   print('$$',self.VNF_list[i].vnf_id)
        for i in range(len(self.VNF_list)):
            # print('**',self.VNF_list[i-result].vnf_id)
            if self.VNF_list[i - result].vnf_id == del_id:
                self.CPU_R += self.VNF_list[i - result].cpu_num
                # print('free num=', self.VNF_list[i-result].CPU_total,'avail num=',self.get_CPU_R())
                del self.VNF_list[i - result]
                result += 1
                if (i - result) >= len(self.VNF_list):
                    break
        return result

    def get_CPU_R(self):
        return self.CPU_R


# define controller node class
class controller:
    def __init__(self, AP_num, C_num):
        self.AP_num = 0
        self.C_num = 0
        self.profit = 0
        self.AP_num = AP_num
        self.C_num = C_num

    def W_allocate(self):  # allocate weights of shared nodes for involved APs
        print("Weight reassigned. ")

    def P_collect(self):  # collect profit of edge network

        return self.profit

    # define directed link class


class D_link:
    def __init__(self, link_id, A_id, B_id, link_map):
        self.link_id = link_id
        self.A_id = A_id  # endpoint A of this link
        self.B_id = B_id  # another endpoint B
        self.unit_cost = 1  # the cost of carrying one unit of flow
        self.Avail = 1  # the availability of this link
        self.P_delay = 10  # the propagation delay of this link in microsecond
        self.T_capacity = 100  # the amount of traffic this link can hold
        self.T_ab = 0  # the traffic of a->b
        self.T_ba = 0  # the traffic of a->b
        self.T_ab_R = self.T_capacity  # residual bandwidth of a->b
        self.T_ba_R = self.T_capacity  # residual bandwidth of b->a
        link_map[A_id, B_id] = link_id
        link_map[B_id, A_id] = link_id
        self.path_list = []  # list of paths which include this link !!!

    def reset(self):
        self.T_ab = 0
        self.T_ba = 0
        self.T_ab_R = self.T_capacity
        self.T_ba_R = self.T_capacity

    def self_report(self):
        print("I am directed link %d from %d to %d, I can hold %d traffics. "
              % (self.link_id, self.A_id, self.B_id, self.T_capacity))

    def state_report(self):
        print("The traffic usage of link %d is: " % (self.link_id))

    # config traffic for link A -> B
    # amount > 0 for adding traffic and amount < 0 for deleting traffic
    def config_traffic(self, A_id, B_id, amount):
        if (A_id == self.A_id) and (B_id == self.B_id):
            if (self.T_capacity >= self.T_ab + amount):
                self.T_ab = self.T_ab + amount
                self.T_ab_R = self.T_ab_R - amount
            else:
                print("request denied, capacity exceeded.")
        elif (A_id == self.B_id) and (B_id == self.A_id):
            if (self.T_capacity >= self.T_ba + amount):
                self.T_ba = self.T_ab + amount
                self.T_ba_R = self.T_ba_R - amount
            else:
                print("request denied, capacity exceeded.")
        else:
            print("link with endpoint (%d, %d) not exists." % (A_id, B_id))

    def get_endpoint(self):
        return self.A_id, self.B_id

    def get_BW_R(self, x, y):  # get residual bandwidth of this link
        if self.A_id == x and self.B_id == y:
            return self.T_ab_R
        elif self.A_id == y and self.B_id == x:
            return self.T_ba_R

    def get_latency(self):  # get latency of this link
        return self.P_delay

    def add_path_ref(self, path_id):  # add path which uses this link
        self.path_list.append(path_id)
        return 0

    def path_update_trigger(self):  # update paths if this link changes it bw usage
        for i in len(self.path_list):
            self.path_list[i].get
