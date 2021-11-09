# used in interations
def check_node_nums(solution1, solution2):
    s1_node_nums = sum([len(rt.node_list) - 1 for rt in solution1.route_list])
    s2_node_nums = sum([len(rt.node_list) - 1 for rt in solution2.route_list])

    return s1_node_nums == s2_node_nums


# for alns, also used in interations
def check_node_nums_before_repair(solution1, solution2):
    if not solution1.relaxed_nodes:
        s1_node_nums = sum([len(rt.node_list) - 1 for rt in solution1.route_list])
    else:
        s1_node_nums = sum([len(rt.node_list) - 1 for rt in solution1.route_list]) + len(solution1.relaxed_nodes)

    if not solution2.relaxed_nodes:
        s2_node_nums = sum([len(rt.node_list) - 1 for rt in solution2.route_list])
    else:
        s2_node_nums = sum([len(rt.node_list) - 1 for rt in solution2.route_list]) + len(solution2.relaxed_nodes)

    print("s1_node_nums", s1_node_nums)
    print("s2_node_nums", s2_node_nums)
    return s1_node_nums == s2_node_nums


# when reading or writing solution, check it by this way
# para: the weight and volume constraint
def check_solution(problem_data, solution, para):
    node_nums = len(problem_data.data)

    # exact number of nodes
    # rt[1:]: remove the starting point
    s_node_nums = sum([len(set(rt[1:])) for rt in solution.route_list])
    # the number of nodes that might include the same point
    s_node_nums_without_set = sum([len(rt[1:]) for rt in solution.route_list])

    # check if the number of nodes is right
    if node_nums == s_node_nums and s_node_nums == s_node_nums_without_set:
        # for each route, check the weight and volume constraint
        for route in solution.route_list:
            weight = sum([problem_data[node_index]["weight"] for node_index in route])
            volume = sum([problem_data[node_index]["volume"] for node_index in route])
            if weight > para.weight or volume > para.volume:
                return False

        return True
    else:
        return False

