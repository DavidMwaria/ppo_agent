# define statistical class for each episode
from typing import Dict


class epi_stat:
    def __init__(self):
        self.epi_id = 0
        self.total_req_num = 0
        self.reject_num = 0
        self.total_reward = 0
        self.fd = None
        self.req_intensity = None

        # count the number of requests processed by each pattern critic network
        self.req_per_critic: Dict[int, int]

    def start_recording(self):
        self.fd = open("exp_result.txt", "w+")

    def episode_reset(self):
        self.epi_id += 1
        self.total_req_num = 0
        self.reject_num = 0
        self.total_reward = 0

    def add_record(self, result, reward):
        self.total_req_num += 1
        if result == 0:
            self.reject_num += 1  # request rejected
            return 1
        else:
            self.total_reward += reward  # add reward for serving request
            return 0

    def get_accept_ratio(self):
        return (self.total_req_num - self.reject_num) / self.total_req_num

    def push_result(self):
        str1 = 'episode ' + str(self.epi_id) + ':\n'
        str2 = 'total request num: ' + str(self.total_req_num) +'\n'
        accept_ratio = self.get_accept_ratio()
        str3 = 'accept ratio: ' + str(accept_ratio) + '\n'
        line = str1+str2+str3
        self.fd.write(line)

    def end_recording(self):
        self.fd.close()
