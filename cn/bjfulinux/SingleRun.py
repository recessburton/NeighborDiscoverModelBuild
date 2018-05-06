#! /usr/bin/env python3
import simpy
import configparser
import random
import time
import math
from cn.bjfulinux.Node import Node
from cn.bjfulinux.SignalMath import SignalMath

# Read Config file: config.ini
conf = configparser.ConfigParser()
conf.read('config.ini')

# Get simulation config
SIM_TIME = conf.get('simulation', 'simulation_hours') * 3600
RADIO_TRANS_TIME = float(conf.get('topo', 'radio_trans_time_milli')) / 1000.0
TRANS_POWER = conf.get('topo', 'trans_power')
MIN_RCV_RSSI = conf.get('topo', 'min_rcv_rssi')  # http://www.docin.com/p-266689616.html
DUTYCYCLE_UPPER = conf.get('simulation', 'dutycycle_upper')
DUTYCYCLE_LOWER = conf.get('simulation', 'dutycycle_lower')
NEIGHBOR_PROBE_INTERVAL_MILLI = float(conf.get('simulation', 'neighbor_probe_interval_milli')) / 1000


def get_node_by_id(nodes, wanted_id):
    for seed_node in nodes:
        if seed_node.nodeid == wanted_id:
            return seed_node


def common_neighbor_num(list1, list2, list3=None):
    if not list3:
        list3 = list2
    return list(set(list1) & set(list2) & set(list3)).__len__()


def relative_distance(distance_from_rssi, comm_nei):
    return distance_from_rssi / float(comm_nei)


def common_neighbor_rate(list1, list2, list3=None):
    """
    :param list1: neighbors of node 1
    :param list2: neighbors of node 2
    :param list3: neighbors of node 3
    :return: Common neighbor rate between node 1, 2 and 3
    e.g. node 1 has neighbors: 3 4 6, node 2 has neighbors 3 4 7 8,
    then common neighbor rate between node 1 and 2 are 2/5.
    """
    if not list3:
        list3 = list2
    return float(list(set(list1) & set(list2) & set(list3)).__len__()) / \
           float(list(set(list1) | set(list2) | set(list3)).__len__())


def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


def distance_in_rssi(rssi):
    """
    Describe distance from rssi side.
    :param rssi:
    :return:
    """
    d_rssi = float(sigmoid(rssi / 10.0 + 5))
    return d_rssi


def distance(node1, node2, common_neighbor_rate_value):
    """
    Novel metric to describe distance between node1 and node2.
    :param node1:
    :param node2:
    :param common_neighbor_rate_value:
    :return:
    """
    accumulate_dis = 0.0
    common_num = 0
    for nodei in node1:
        if nodei in node2.keys():
            common_num += 1
            accumulate_dis += distance_in_rssi((node2[nodei] + node1[nodei]) / 2.0)
    if common_num == 0:
        return 0
    avg_dis = accumulate_dis / common_num
    return round(math.sqrt(avg_dis * common_neighbor_rate_value), 4)


def sort_neighbors_by_rssi(node):
    neighbors_list = [node.neighbors[key]['rssi'] for key in node.neighbors]
    return sorted(neighbors_list, reverse=True)


def log_recommend(referer, presentee, target, shouldrefer, noise_thre, trans_power, min_rssi):
    if presentee.nodeid not in referer.neighbors.keys():
        return
    if target.nodeid not in referer.neighbors.keys():
        return

    # form neighbor list
    referer_neighbors = {}
    for nei_node_id in referer.neighbors.keys():
        referer_neighbors[nei_node_id] = referer.neighbors[nei_node_id]['rssi']
    presentee_neighbors = referer.neighbors[presentee.nodeid]['sub_neighbors']
    target_neighbors = referer.neighbors[target.nodeid]['sub_neighbors']
    referer_neighbors_list = list(referer.neighbors.keys())
    referer_neighbors_list.append(referer.nodeid)
    presentee_neighbors_list = list(presentee_neighbors.keys())
    presentee_neighbors_list.append(presentee.nodeid)
    target_neighbors_list = list(target_neighbors.keys())
    target_neighbors_list.append(target.nodeid)

    # common neighbor rate metrics
    common_neighbor_r_p = common_neighbor_rate(referer_neighbors_list, presentee_neighbors_list)
    common_neighbor_r_t = common_neighbor_rate(referer_neighbors_list, target_neighbors_list)
    common_neighbor_p_t = common_neighbor_rate(presentee_neighbors_list, target_neighbors_list)
    if not (common_neighbor_p_t and common_neighbor_r_p and common_neighbor_r_t):
        return

    # distance metrics
    distance_r_p = distance(referer_neighbors, presentee_neighbors, common_neighbor_r_p)
    distance_r_t = distance(referer_neighbors, target_neighbors, common_neighbor_r_t)
    # distance_p_t_debug = distance(presentee_neighbors, target_neighbors, common_neighbor_p_t)

    if not (distance_r_p and distance_r_t):
        return

    # common neighbor rate between RP and RT, i.e. common neighbors between RPT
    common_neighbor_rate_trilateral = common_neighbor_rate(referer_neighbors_list, presentee_neighbors_list,
                                                           target_neighbors_list)

    # calculate distance from rssi
    distance_from_rssi_r_p = SignalMath.resume_distance_from_rssi(referer_neighbors[presentee.nodeid],
                                                                  float(trans_power))
    distance_from_rssi_r_t = SignalMath.resume_distance_from_rssi(referer_neighbors[target.nodeid],
                                                                  float(trans_power))

    # rate between distance and common neighbor num
    k1 = SignalMath.square_in_common_nei(trans_power, min_rssi, distance_from_rssi_r_p) \
         / float(common_neighbor_num(referer_neighbors_list, presentee_neighbors_list))
    k2 = SignalMath.square_in_common_nei(trans_power, min_rssi, distance_from_rssi_r_t) \
         / float(common_neighbor_num(referer_neighbors_list, target_neighbors_list))

    # common square between P and T
    square = math.sqrt((k1 ** 2 + k2 ** 2)/2) * common_neighbor_p_t

    # distance between p and t
    distance_p_t = SignalMath.solve_d_from_square(square, trans_power, min_rssi)

    # top n rssi of neighbors
    sorted_neighbors_of_presentee = sort_neighbors_by_rssi(presentee)
    average_rssi_of_presentee = round(sum(sorted_neighbors_of_presentee.__iter__()) * 1.0 \
                                      / sorted_neighbors_of_presentee.__len__(), 2)

    # write log
    with open('neighbor-log.txt', 'a+') as log:
        print(distance_r_p, distance_r_t, round(distance_p_t, 4), referer_neighbors[presentee.nodeid],
              referer_neighbors[target.nodeid], round(common_neighbor_r_p, 4), round(common_neighbor_r_t, 4),
              round(common_neighbor_p_t, 4), round(common_neighbor_rate_trilateral, 4),
              sorted_neighbors_of_presentee[0], str(average_rssi_of_presentee), noise_thre, str(shouldrefer), file=log)


def can_hear_each_other(node1_str, node2_str, env):
    node1 = get_node_by_id(env.nodes, node1_str)
    node2 = get_node_by_id(env.nodes, node2_str)
    rssi = SignalMath.cal_rssi(node1, node2, env.TRANS_POWER)
    return 0 if rssi < float(env.MIN_RCV_RSSI) else 1


def run_simulation(referer_node='1', target_node='2', noise_thre=2):
    # WSN Environment
    env = simpy.Environment()
    env.wireless_channel = simpy.Resource(env, capacity=1)  # Radio Channel, Only one node can talk at a time
    env.nodes = []
    env.RADIO_TRANS_TIME = RADIO_TRANS_TIME
    env.TRANS_POWER = TRANS_POWER
    env.MIN_RCV_RSSI = MIN_RCV_RSSI
    env.NEIGHBOR_PROBE_INTERVAL_MILLI = NEIGHBOR_PROBE_INTERVAL_MILLI
    env.noise_thre = noise_thre
    env.target_node = str(target_node)
    referer_node = str(referer_node)

    # load nodes from nodes.txt
    with open('nodes.txt') as nodes_conf:
        line = nodes_conf.readline()
        while line and line != "\n":
            node_conf = line.split(' ')
            random.seed(time.time() + random.random())
            dutycycle = random.randint(int(DUTYCYCLE_LOWER), int(DUTYCYCLE_UPPER))
            node = Node(env, node_conf[0], node_conf[1], node_conf[2].strip(), TRANS_POWER, dutycycle)
            env.nodes.append(node)
            line = nodes_conf.readline()

        # add random nodes in region
        for nodeid in range(3, 40):
            random.seed(time.time() + random.random())
            dutycycle = random.randint(int(DUTYCYCLE_LOWER), int(DUTYCYCLE_UPPER))
            x = random.random() * 27 - 7  # x in [-7, 20]
            y = random.random() * 25 - 10  # y in [-10, 15]
            node = Node(env, str(nodeid), str(x), str(y), TRANS_POWER, dutycycle)
            env.nodes.append(node)

    # Run simulation
    env.run(until=3)

    # referer: 1, presentee: 101, target: target_node
    should_refer = can_hear_each_other('101', env.target_node, env)
    log_recommend(get_node_by_id(env.nodes, referer_node), get_node_by_id(env.nodes, '101'),
                  get_node_by_id(env.nodes, env.target_node), should_refer, noise_thre, TRANS_POWER, MIN_RCV_RSSI)
    # referer: 1, presentee: 102, target: target_node
    should_refer = can_hear_each_other('102', env.target_node, env)
    log_recommend(get_node_by_id(env.nodes, referer_node), get_node_by_id(env.nodes, '102'),
                  get_node_by_id(env.nodes, env.target_node), should_refer, noise_thre, TRANS_POWER, MIN_RCV_RSSI)


if __name__ == '__main__':
    run_simulation()
