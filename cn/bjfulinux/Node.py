# -*- encoding: utf-8 -*-
from cn.bjfulinux.Message import Message
import random
import time


class Node:
    def __init__(self, env, nodeid, x, y, power, dutycycle):
        self.neighbors = {}  # {nodeid:{rssi:rssi, dutycycle:dutycycle, sub_neighbors:{nodeid1:rssi1, nodeid2:rssi2,...}}
        self.active = False
        self.nodeid, self.x, self.y, self.power, self.dutycycle = nodeid, x, y, power, dutycycle
        #print("%s node %s deployed at (%s, %s) with power %s, dutycycle: %s%%"
        #      % (env.now, nodeid, x, y, power, dutycycle))
        env.process(self.random_boot(env))

    def random_boot(self, env):
        random.seed(time.time() + random.random())
        yield env.timeout(float(random.randint(0, 1000)) / 1000.0)
        #print("%s node %s booted" % (env.now, self.nodeid))
        self.active = True
        env.process(self.start_dutycycle(env))

    def start_dutycycle(self, env):
        while True:
            env.process(self.send_neighbor_probe(env))
            yield env.timeout(self.dutycycle / 100)
            #print("%s node %s sleep" % (env.now, self.nodeid))
            self.active = False
            yield env.timeout(1 - self.dutycycle / 100)
            #print("%s node %s active" % (env.now, self.nodeid))
            self.active = True

    def send_neighbor_probe(self, env):
        """
        :param env: environment
        :return: none
        Send neighbor discover message. Ignore message collision here.
        """
        while self.active:
            """ ****ignore***
            with env.wireless_channel.request() as channel:
                yield channel
                print("%s node %s get the channel and send probe" % (env.now, self.nodeid))
                # create a radio event
                message = Message(self)
                message.should_receive(env)
                yield env.timeout(env.RADIO_TRANS_TIME)
            """
#            print("%s node %s send probe" % (env.now, self.nodeid))
            # create a radio event
            message = Message(self)
            message.should_receive(env)
            yield env.timeout(env.RADIO_TRANS_TIME)
            yield env.timeout(env.NEIGHBOR_PROBE_INTERVAL_MILLI)

    def receive(self, env, message, rssi):
        if not self.active:
            return
        if message.source_node.nodeid not in self.neighbors.keys():
            pass  # print("%s node %s find neighbor %s with rssi %.2f" % (env.now, self.nodeid, message.source_node.nodeid, rssi))
        self.neighbors[message.source_node.nodeid] = {}
        self.neighbors[message.source_node.nodeid]['rssi'] = rssi
        self.neighbors[message.source_node.nodeid]['dutycycle'] = message.source_node.dutycycle
        self.neighbors[message.source_node.nodeid]['sub_neighbors'] = {}
        for neighbor_id, other_info_dict in message.source_node.neighbors.items():
            self.neighbors[message.source_node.nodeid]['sub_neighbors'][neighbor_id] = other_info_dict['rssi']
        #print("%s node %s: " % (env.now, self.nodeid), self.neighbors)


