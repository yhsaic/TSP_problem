from sklearn.cluster import DBSCAN
from math import radians, asin, sqrt, pow, sin, cos
from models.route import Route
import numpy as np
from models.solution import Solution


def cluster_construct_algorithm(center_node, unserved_node_list, distance, sample_size):
    cluster_nodes_dict, noise_nodes_dict, labels = get_clusters(unserved_node_list, distance, sample_size)

    # for nodes in cluster
    routes = []
    for value in cluster_nodes_dict.keys():
        tmp_route = Route()
        tmp_route.vehicle_ID = value
        tmp_route.node_list = [center_node] + [unserved_node_list[index] for index in cluster_nodes_dict[value]]
        tmp_route.volume = np.sum([unserved_node_list[index].volume for index in cluster_nodes_dict[value]])
        tmp_route.weight = np.sum([unserved_node_list[index].weight for index in cluster_nodes_dict[value]])
        routes.append(tmp_route)

    # for noise nodes
    tmp_route = Route()
    tmp_route.vehicle_ID = len(cluster_nodes_dict.keys())
    tmp_route.node_list = [center_node] + [unserved_node_list[index] for index in noise_nodes_dict[-1]]
    tmp_route.volume = np.sum([unserved_node_list[index].volume for index in noise_nodes_dict[-1]])
    tmp_route.weight = np.sum([unserved_node_list[index].weight for index in noise_nodes_dict[-1]])
    routes.append(tmp_route)

    initial_solution = Solution()
    initial_solution.route_list = routes
    return initial_solution


# 只是针对所有的nodes级别进行cluster
def get_clusters(unserved_node_list, distance, sample_size):
    points = []
    for node in unserved_node_list:
        if node.node_type != 'center':
            points.append([node.lng, node.lat])

    # eps: 一个核心点可以覆盖的半径，单位米
    # min_samples：一个簇最少由几个点组成
    # metric：距离的计算方式

    sample_size = sample_size  # 为了防止route本身包含的nodes数过少，而导致全部都是噪音的情况，min_samples的值需要动态调整
    if len(unserved_node_list) - 1 < sample_size:
        sample_size = len(unserved_node_list) - 1

    dbscan = DBSCAN(eps=distance, min_samples=sample_size, metric=get_distance).fit(points)

    # 分析labels：标记为-1的表示噪音，除此之外，0, 1, 2, ... 表示不同的组
    # dbscan.labels_的格式是一个list, 长度等于points列表的长度，对应位置的值表示对应的点所属的cluster
    cluster_index_dict = {}  # {-1: [1, 3], 0:[2, 4, 11], 1: [5, 8], 2: [x, x, x, ...]}  key为cluster，value为属于该组的点的下标
    cluster_nodes_dict = {}
    noise_nodes_dict = {}

    for index, value in enumerate(dbscan.labels_):
        if value != -1:
            if value not in cluster_nodes_dict.keys():
                cluster_nodes_dict[value] = [index]
            else:
                cluster_nodes_dict[value].append(index)
        else:
            if value not in noise_nodes_dict.keys():
                noise_nodes_dict[value] = [index]
            else:
                noise_nodes_dict[value].append(index)

    return cluster_nodes_dict, noise_nodes_dict, dbscan.labels_


# 通过坐标计算地图上的距离
def get_distance(array_1, array_2):
    lon_a = array_1[0]
    lat_a = array_1[1]
    lon_b = array_2[0]
    lat_b = array_2[1]
    radlat1 = radians(lat_a)
    radlat2 = radians(lat_b)
    a = radlat1 - radlat2
    b = radians(lon_a) - radians(lon_b)
    s = 2 * asin(sqrt(pow(sin(a / 2), 2) + cos(radlat1) * cos(radlat2) * pow(sin(b / 2), 2)))
    earth_radius = 6378137
    s = s * earth_radius
    return s
