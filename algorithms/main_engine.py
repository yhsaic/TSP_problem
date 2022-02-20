import json
from config.parameters import Para
from models.node import Node
from read_problem_data import read_problem_data
from construct_algorithms.cluster import cluster_construct_algorithm
from iteration_algorithms.ALNS import alns

# read instance data
ori_data_path = "./database/2020-11-07上午_data_with_lnglat.txt"
encoding = "GBK"
problem_data = read_problem_data(ori_data_path, encoding)

# read distance data
write_path = "../database/2020-11-07上午_distance.txt"
distance_dict = {}
with open(write_path, 'r', encoding=encoding) as f:
    distance_dict = json.load(f)

# define center node
para = Para()
center_node = Node()
center_node.id = 0
center_node.type = 'center'
center_node.lng = para.center_node_lnglat[0]
center_node.lat = para.center_node_lnglat[1]

# define other nodes
unserviced_nodes = []
data = problem_data.data
for i in range(0, len(data)):
    tmp_node = Node()
    tmp_node.id = i + 1
    tmp_node.type = 'normal'
    tmp_node.weight = data[i]["weight"]
    tmp_node.volume = data[i]["volume"]
    tmp_node.lng = data[i]["lnglat"][0]
    tmp_node.lat = data[i]["lnglat"][1]

    unserviced_nodes.append(tmp_node)

distance = 10
sample_size = 3

# nearest neighbor, cheapest insertion, random insertion, cluster?
initial_solution = cluster_construct_algorithm(center_node, unserviced_nodes, distance, sample_size)

final_solution = alns(initial_solution, para, center_node, distance_dict, data)

# solution = ALNS()

# print the result
