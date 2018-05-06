# -*- encoding: utf-8 -*-
import uuid
from cn.bjfulinux.SignalMath import SignalMath


class Message:

    def __init__(self, node):
        self.uuid = uuid.uuid1()
        self.source_node = node

    def should_receive(self, env):
        for node in env.nodes:
            # 101 and 102 are target node
            if node.nodeid == env.target_node:
                if self.source_node.nodeid == '102' or self.source_node.nodeid == '101':
                    continue
            if node.nodeid == '101' or node.nodeid == '102':
                if self.source_node.nodeid == env.target_node:
                    continue

            if node.nodeid == self.source_node.nodeid:
                continue
            else:
                rssi = SignalMath.cal_rssi(self.source_node, node, env.TRANS_POWER, env.noise_thre)
                if rssi < float(env.MIN_RCV_RSSI):
                    continue  # can not receive the message
                else:  # receive
                    node.receive(env, self, rssi)
