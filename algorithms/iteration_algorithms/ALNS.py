import time
import math
import os

from copy import deepcopy
import numpy as np
import random
import json
import multiprocessing as mp

from models.solution import Solution
from sklearn.cluster import KMeans


def alns(init_solution, para, center_node, distance_dict, data):
    # 查看自适应参数k和non_improve_count在迭代过程中的变化
    non_improve_count_list = []
    k_list = []

    print("objective: %s || total_violation_penalty: %s || intersected_violation: %s" % (init_solution.objective, init_solution.total_violation_penalty, init_solution.intersected_violation))
    print("imbalance_penalty: %s || silhouette_score: %s || cost: %s || vehicle_to_route_penalty: %s" % (init_solution.imbalance_penalty, init_solution.silhouette_score, init_solution.cost, init_solution.vehicle_to_route_penalty))
    print("capacity_violation: %s || time_violation: %s || distance_violation: %s || duration_violation: %s" % (init_solution.capacity_violation, init_solution.time_violation, init_solution.distance_violation, init_solution.duration_violation))
    print("interval_violation: %s || angle_violation: %s || node_number_violation: %s || duration_violation: %s" % (init_solution.interval_violation, init_solution.angle_violation, init_solution.node_number_violation, init_solution.duration_violation))

    print("origin solution objective", init_solution.objective)

    start_time = time.time()
    print("start time", start_time)
    total_iteration_count = non_improve_count = 1
    reorder_count = 0
    current = best = init_solution
    last_iteration_s = init_solution
    # segment_size = 100  # 每100个iteration作为一个segment.一个segment结束后调整一次weights
    segment_size = 15
    d_weights, d_scores, d_trials = initialize_destroy_dicts()
    r_weights, r_scores, r_trials = initialize_repair_dicts()
    cur_temp, cool_rate = simulated_annealing_setting(init_solution)
    k0, k_interval = initialize_k_parameters()

    while terminate(start_time, total_iteration_count, non_improve_count, para):
        # 不计入总循环中
        # 用tabu算法那样理路顺，概率低，但是会缕，而且方案多样化（不过速度慢，也不一定能保证满足时间窗约束）
        re_orders_solutions = []
        # rand = np.random.random()
        # print("rand", rand)
        # if rand < 0.5:
        if True:
            # print("before reorder, current.objective = ", current.objective)
            # reorder_count += 1
            #
            # # reorder算子
            # q = mp.Queue()
            # # re_order_routes(ori_solution, scenario, tabu_dict, aspi_dict, total_iteration_count, q, para)
            # # 其中tabu_dict, aspi_dict, total_iteration_count没有用到，可随便传值
            # re_order_routes(current, scenario, {}, {}, 10, q, para)
            # print("for reorder routes, q.qsize()", q.qsize())
            #
            # while q.qsize():
            #     re_orders_solutions += q.get()

            # 最近邻法
            nearest_reorder_s = deepcopy(current)
            for route in nearest_reorder_s.route_list:
                sort_node_list(route)

            re_orders_solutions.append(nearest_reorder_s)

        print("total_iteration_count", total_iteration_count)
        if total_iteration_count % segment_size == 0:  # 阶段性地调整operator被选中的概率
            adjust_weights(d_weights, d_scores, d_trials, r_weights, r_scores, r_trials)
            clear_for_new_segment(d_scores, d_trials, r_scores, r_trials)  # 计算完成后，为新的segment重置这些字典的值

        # 阶段性地调整k0值
        if non_improve_count_list:
            k0 = adjust_k0(k0, k_interval, non_improve_count_list, total_iteration_count)
        print("k0 = ", k0)

        d = choose_destroy_operator(d_weights)
        # r = choose_repair_operator(r_weights)
        d_trials[d] += 1  # 调用次数累计
        # r_trials[r] += 1  # 调用次数累计

        q = mp.Queue()

        destroy(current, d, scenario, para, q, k0)
        print("q.qsize()", q.qsize())

        neighbor_list = []
        while q.qsize():
            neighbor_list += q.get()
        print("len(neighbor_list)", len(neighbor_list))

        # 工作量均衡的removal和insertion是对应的，而其他的可以任意组合
        # 目前：4*2+1*1=9
        if d == 'workload_based':
            r = 'workload_based_insertion'
        else:
            r = choose_repair_operator(r_weights)

        r_trials[r] += 1  # 调用次数累计

        improve_flag = False
        best_flag = False
        accept_flag = False
        for neighbor in neighbor_list:
            # 保证在拆路线时没有少点
            print("check tmp_solution")
            if check_node_nums_before_repair_new(neighbor, init_solution):
                # print("check tem_solution success")
                new_s = repair(neighbor, r, scenario, para)
                # print("objective: %s || total_violation_penalty: %s || intersected_violation: %s" % (new_s.objective, new_s.total_violation_penalty, new_s.intersected_violation))
                # print("imbalance_penalty: %s || silhouette_score: %s || cost: %s || vehicle_to_route_penalty: %s" % (new_s.imbalance_penalty, new_s.silhouette_score, new_s.cost, new_s.vehicle_to_route_penalty))
                # print("")
                print("check repair solution")
                if check_node_nums_before_repair_new(neighbor, new_s):
                    # # # 判断这次的目标函数值是否有提升
                    # if new_s.objective < last_iteration_s.objective:
                    #     improve_flag = True
                    print("check solution success")
                    if accept_solution(new_s, current, cur_temp):
                        current = new_s
                        accept_flag = True
                        print("accept new solution")
                        # case = 'accept_better_than_current'
                    else:
                        print('rejected')
                        # case = 'reject'

                    if new_s.objective < best.objective:
                        best_flag = True
                        best = new_s
                        # case = 'update_global_best'  # 会覆盖'accept_better_than_current'，从而只会进行一次adjust_scores()操作

                    print("new_s.objective", new_s.objective)
                    print("new_s.alns_operators", new_s.alns_operators)
                    # print("")
                else:
                    print("----------------------------------------")
                    print("check repaired_solution fail")

            else:
                print("----------------------------------------")
                print("check tem_solution fail")

            print("")

        # 得到本次迭代的方案后，尝试做一下工作量均衡处理，目标函数都变大时取本次迭代方案
        # 效果不好，之后加了工作量均衡算子，这部分就更不需要了
        # current = workload_balance_operator(current, scenario, para)
        # print("current.objective after workload", current.objective)
        # if current.objective < best.objective:
        #     best_flag = True
        #     best = current

        # 无法约束函数，使得一定满足时间窗约束
        # 每次迭代后，以一定概率重排一次当前方案的路线（最近邻）
        # rand = np.random.random()
        # if rand < 0.5:
        #     for route in current.route_list:
        #         sort_node_list(route)
        #         # route.status = 'changed'
        #     # evaluate_solution(current, scenario, para)

        # 用reorder方法去修改current和best方案的路线
        if len(re_orders_solutions) > 0:
            current_uuids = [tp_rt.uuid for tp_rt in current.route_list]
            # routes_uuids_dict ={} ##'key:', uuids, values={'sol_index': , 'rt_index': 'rt_distance': }
            for tmp_solution in re_orders_solutions:
                for rt in tmp_solution.route_list:
                    if rt.uuid in current_uuids:
                        rt_solution_index = current_uuids.index(rt.uuid)
                        if rt.total_distance < current.route_list[rt_solution_index].total_distance:
                            current.route_list[rt_solution_index] = rt

        if len(re_orders_solutions) > 0:
            best_uuids = [tp_rt.uuid for tp_rt in best.route_list]
            # routes_uuids_dict ={} ##'key:', uuids, values={'sol_index': , 'rt_index': 'rt_distance': }
            for tmp_solution in re_orders_solutions:
                for rt in tmp_solution.route_list:
                    if rt.uuid in best_uuids:
                        rt_solution_index = best_uuids.index(rt.uuid)
                        if rt.total_distance < best.route_list[rt_solution_index].total_distance:
                            best.route_list[rt_solution_index] = rt

        # 判断这次迭代的目标函数值是否有提升
        if current.objective < last_iteration_s.objective:
            improve_flag = True

        # 增加权重
        obj_difference = 0
        if best_flag:
            case = 'update_global_best'
            obj_difference = (last_iteration_s.objective - best.objective)/100
        elif current.objective < last_iteration_s.objective:
            case = 'accept_better_than_current'
            obj_difference = (last_iteration_s.objective - current.objective)/100
        elif accept_flag:
            case = 'accept_worse_than_current'
        else:
            case = 'reject'

        # 有可能本次循环没接受解，此时将operator设为上一次迭代的operator
        if not current.alns_operators:
            current.alns_operators = last_iteration_s.alns_operators

        print("for one iteration, case = ", case)
        print("current.alns_operators", current.alns_operators)
        print("objective: %s || total_violation_penalty: %s || intersected_violation: %s" % (best.objective, best.total_violation_penalty, best.intersected_violation))
        print("imbalance_penalty: %s || silhouette_score: %s || cost: %s || vehicle_to_route_penalty: %s" % (best.imbalance_penalty, best.silhouette_score, best.cost, best.vehicle_to_route_penalty))
        print("capacity_violation: %s || time_violation: %s || distance_violation: %s || duration_violation: %s" % (best.capacity_violation, best.time_violation, best.distance_violation, best.duration_violation))
        print("interval_violation: %s || angle_violation: %s || node_number_violation: %s || duration_violation: %s" % (best.interval_violation, best.angle_violation, best.node_number_violation, best.duration_violation))

        # 每一个iteration需要做的一些处理
        adjust_scores(current, d_scores, r_scores, case, obj_difference)  # 为d和r累加分数

        last_iteration_s = deepcopy(current)
        # 重置alns_operators，以便下一次迭代(append)
        current.alns_operators = []

        print("d", d)
        print("d_scores[key]", d_scores[d])
        print("r_scores[key]", r_scores[r])

        # 以一定的概率接受当前解，其余情况下取最优解
        # 多试几次，看效果如何
        # rand = np.random.random()
        # if not rand < math.exp((current.objective - best.objective) / cur_temp):
        #     current = best

        # cur_temp /= cool_rate  # 冷却温度
        cur_temp *= cool_rate  # 更改版
        print("cur_temp", cur_temp)
        total_iteration_count += 1
        # 如果本次循环有更好的解，non_improve_count重置为0，否则加1
        # if not improve_flag and not best_flag:
        if not best_flag:
            non_improve_count += 1
        else:
            non_improve_count = 0

        k_list.append(k0)
        non_improve_count_list.append(non_improve_count)

        print("non_improve_count", non_improve_count)
        print("best.objective", best.objective)
        print("")

    print("reorder_count", reorder_count)

    # 发掘设置自适应k过程
    print("total_iteration_count", total_iteration_count)
    print("-------- k_list ------", k_list)
    print("non_improve_count_list", non_improve_count_list)

    start_datetime = 0
    for route in best.route_list:
        start_transport_time = None
        for node in route.node_list:
            if node.node_type != 'dc' and node.transport_time:
                start_transport_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime((int(node.transport_time)) / 1000))
                break
        start_datetime = date_str_to_minutes(start_transport_time)  # 开始运输的时间  发车时间
        if start_datetime:
            break

    print("start_datetime", start_datetime)

    # 读取json的全部内容，然后再在后面添加东西
    d = os.path.dirname(__file__)
    # print("d", d)
    # path = os.getcwd() + '/Database/json_solution.json'  tw_solution_record change_parameter
    path = d + '/Database/change_parameter.json'
    print("path", path)
    with open(path, 'r', encoding='utf8')as fp:
        load_dict = json.load(fp)
        print("load_dict", load_dict)

    return best


def initialize_destroy_dicts():
    d_weights = {'random_node': 1, 'node_sil_value': 1, 'random_route': 1, 'route_obj_based': 1, 'workload_based': 1}
    d_scores = {'random_node': 0, 'node_sil_value': 0, 'random_route': 0, 'route_obj_based': 0, 'workload_based': 0}
    d_trials = {'random_node': 0, 'node_sil_value': 0, 'random_route': 0, 'route_obj_based': 0, 'workload_based': 0}
    # d_weights = {'workload_based': 1}
    # d_scores = {'workload_based': 0}
    # d_trials = {'workload_based': 0}
    return d_weights, d_scores, d_trials


def initialize_repair_dicts():
    r_weights = {'dist_based_insertion': 1, 'dist_based_insertion_cluster': 1, 'workload_based_insertion': 1, 'merge_and_split_insertion': 1}
    r_scores = {'dist_based_insertion': 0, 'dist_based_insertion_cluster': 0, 'workload_based_insertion': 0, 'merge_and_split_insertion': 0}
    r_trials = {'dist_based_insertion': 0, 'dist_based_insertion_cluster': 0, 'workload_based_insertion': 0, 'merge_and_split_insertion': 0}
    # r_weights = {'merge_and_split_insertion': 1}
    # r_scores = {'merge_and_split_insertion': 0}
    # r_trials = {'merge_and_split_insertion': 0}
    return r_weights, r_scores, r_trials


def simulated_annealing_setting(init_solution):
    # 为SA选定合适的参数 (ref: Ropke, Pisinger, 2005 - TS)
    # 初始温度选择的原则，是比初始解差5%的解被接受的概率为50%。推导的过程省略
    start_temp = - 0.05 * init_solution.objective / math.log(0.5)  # log不指定底数时，默认底数为e
    # start_temp = 100
    # 冷却系数，cool temperature every iteration
    # cool_rate = 0.99975
    cool_rate = 0.95
    return start_temp, cool_rate


def initialize_k_parameters():
    # 选定算法搜索范围和扩大减小步伐
    k0 = 10  # 初始值，destroy的max_node = total_node_num / k0，决定算法搜索范围
    k_interval = 1  # 自适应值，即在k的基础上增加/减少多少
    return k0, k_interval


def accept_solution(new, current, temp):
    # 基于某种规则，判断是否要接受new solution作为新的work solution
    # 这里暂时采用的是Simulated Annealing式的Acceptance criterion

    if new.objective < current.objective:  # SA中，好的解一定被接受
        return 1

    else:  # 差的解以一定概率接受
        rand = np.random.random()
        # p0 = 1/(1 + e ^( -0.025 * T)) - 1/2
        # T0 = 100, tend = 0.1
        # rate = 0.96
        print("rand", rand)
        print("math.exp((current.objective - new.objective) / temp)", math.exp((current.objective - new.objective) / temp))
        if rand < math.exp(100 * (current.objective - new.objective) / temp):  # TODO 控制量纲一致
            return 1
        else:
            return 0


def choose_destroy_operator(d_weights):
    # 选择一个destroy operator
    sorted_list = sorted(d_weights.items(), key=lambda x: x[1])  # 其实无所谓升降序，只是转化成固定顺序的list即可
    values = []
    names = []
    for item in sorted_list:
        values.append(item[1])
        names.append(item[0])

    index = 0
    values = values / np.sum(values)
    rand_p = random.random()
    cumulated_p = 0
    for i in range(0, len(values)):
        cumulated_p += values[i]
        if cumulated_p > rand_p:
            index = i
            break

    print('chosen destroy operator:', names[index])
    return names[index]


def choose_repair_operator(r_weights):
    # 选择一个repair operator

    sorted_list = sorted(r_weights.items(), key=lambda x: x[1])  # 其实无所谓升降序，只是转化成固定顺序的list即可
    values = []
    names = []
    for item in sorted_list:
        # 对不是工作量平衡的其他算子，根据权重选择
        if item[0] != 'workload_based_insertion':
            values.append(item[1])
            names.append(item[0])

    index = 0
    values = values / np.sum(values)
    rand_p = random.random()
    cumulated_p = 0
    for i in range(0, len(values)):
        cumulated_p += values[i]
        if cumulated_p > rand_p:
            index = i
            break

    print('chosen repair operator:', names[index])
    return names[index]  # 前者是operator下标，后者就是operator name (key)


def destroy(s, d, sce, para, q, k0):
    total_node_num = sum([len(r.node_list)-1 for r in s.route_list])
    node_route_ratio = 2  # 当remove_num太大时，把remove的随机点数转化为路线数的比例 一开始是4

    # min_remove = 4
    # max_remove = min([100, 0.4 * len(sce.demanding_nodes_dict)])
    max_remove = int(total_node_num/k0)
    min_remove = min(4, int(max_remove/2))
    # max_remove = 25  # 30 20
    print("max_remove", max_remove)
    print("min_remove", min_remove)
    remove_num = np.random.randint(min_remove, max_remove)  # TODO: 依据的是《An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows》，需要进一步tuning
    print("remove_num", remove_num)

    if d == 'random_node':
        random_node_based_removal(s, sce, para, remove_num, q)

    elif d == 'node_sil_value':
        node_sil_based_removal(s, sce, para, remove_num, q)

    elif d == 'random_route':
        remove_num = remove_num if remove_num <= len(s.route_list) else remove_num / node_route_ratio
        random_route_based_removal(s, remove_num, q)

    elif d == 'route_obj_based':
        remove_num = remove_num if remove_num <= len(s.route_list) else remove_num / node_route_ratio
        route_obj_based_removal(s, remove_num, q)

    elif d == 'workload_based':
        workload_balance_removal(s, sce, q, para)

    # 如果需要evaluate route/solution, 才需要sce
    # elif d == 'worst':
    #     res_s = worst_removal(s, remove_num, sce)


def repair(s, r, sce, para):
    # print("repair part")
    res_s = s
    if r == 'dist_based_insertion':
        res_s = dist_based_insertion(s, sce, para)

    elif r == 'dist_based_insertion_cluster':
        res_s = dist_based_insertion_cluster(s, sce, para)

    elif r == 'workload_based_insertion':
        res_s.alns_operators.append('workload_based')
        res_s.alns_operators.append('workload_based_insertion')
        print("res_s.alns_operators", res_s.alns_operators)

    elif r == 'merge_and_split_insertion':
        res_s = merge_and_split_insertion(s, sce, para)

    return res_s


# TODO 未调试，也不确定
def adjust_k0(k0, k_interval, non_improve_count_list, total_iteration_count):
    # k0的取值范围为[min_k, max_k]，即取1/max_k-1/min_k的点
    max_k = 15
    min_k = 5
    k_segment = 5

    # 根据non_improvement，调整k0值
    non_improve_count = non_improve_count_list[-1]
    if non_improve_count >= 3:
        k0 -= 2 * k_interval
    # 五次里四次都是0，缩小搜索范围
    elif total_iteration_count % k_segment == 0:
        latest_5_counts = non_improve_count_list[-5:]
        zero_count = latest_5_counts.count(0)
        if zero_count >= 4:
            k0 += k_interval

    k0 = max(min(k0, max_k), min_k)  # 保持k0的取值在范围内

    return k0


def adjust_scores(s, d_scores, r_scores, case, obj_difference):
    # 根据表现，调整各operator的scores
    found_best = 5  # 更新了global best solution (6)
    accept_better_than_current = 2  # better than current working solution (3)
    accept_worst_than_current = 1  # worse than current working solution, but accepted by SA (1)

    d = s.alns_operators[0]
    r = s.alns_operators[1]

    obj_difference = max(1, obj_difference)
    # obj_difference = max(1, min(20, obj_difference))   # 防止最先操作的算子拥有远比其他算子高的权重（5不大好用）
    # 权重的调整问题：5、10、20、50？感觉不调整就很好

    if case == 'update_global_best':
        d_scores[d] += found_best * obj_difference
        r_scores[r] += found_best * obj_difference

    elif case == 'accept_better_than_current':
        d_scores[d] += accept_better_than_current * obj_difference
        r_scores[r] += accept_better_than_current * obj_difference

    elif case == 'accept_worse_than_current':
        d_scores[d] += accept_worst_than_current
        r_scores[r] += accept_worst_than_current

    elif case == 'reject':
        pass


def adjust_weights(d_weights, d_scores, d_trials, r_weights, r_scores, r_trials):
    reaction_parameter = 0.1

    """
    The reaction factor r controls how quickly the weight adjustment algorithm reacts 
    to changes in the effectiveness of the heuristics. 
    If r is zero then we do not use the scores at all 
    and stick to the initial weights. If r is set to one 
    then we let the score obtained in the last segment decide the weight.
    """

    for key in d_weights.keys():
        # 没有调用的算子权重不变
        if d_trials[key] != 0:
            print("d_weights[key] before", d_weights[key])
            d_weights[key] = d_weights[key] * (1 - reaction_parameter) + reaction_parameter * d_scores[key] / d_trials[key]
            print("d_weights[key] after", d_weights[key])
        else:
            print("d_weights[key]", d_weights[key])

    for key in r_weights.keys():
        if r_trials[key] != 0:
            print("r_weights[key] before", r_weights[key])
            r_weights[key] = r_weights[key] * (1 - reaction_parameter) + reaction_parameter * r_scores[key] / r_trials[key]
            print("r_weights[key] after", r_weights[key])
        else:
            print("r_weights[key]", r_weights[key])


def clear_for_new_segment(d_scores, d_trials, r_scores, r_trials):
    # 每一个新的segment开始之前都要重置这些字典
    for key in d_scores.keys():
        d_scores[key] = 0

    for key in d_trials.keys():
        d_trials[key] = 0

    for key in r_scores.keys():
        r_scores[key] = 0

    for key in r_trials.keys():
        r_trials[key] = 0


def terminate(start_time, total_iteration_count, non_improve_count, para):
    # 任何一个指标超过循环上限，则停止计算，返回1
    if (time.time() - start_time) > para.CPU_limit * 60:
        print("Maximum CPU time reached. [Now: %.1f seconds | Limit: %.1f seconds]" % (time.time() - start_time, para.CPU_limit * 60))
        return 0

    if total_iteration_count >= para.total_iteration_limit:
        print("Maximum total iteration reached. [Now: %d iterations | Limit: %d iterations]" % (total_iteration_count, para.total_iteration_limit))
        return 0

    if non_improve_count >= para.tabu_non_improvement_limit:
        print("Maximum non-improve iteration reached. [Now: %d iterations | Limit: %d iterations]" % (non_improve_count, para.tabu_non_improvement_limit))
        return 0

    else:
        return 1  # 否则继续循环
