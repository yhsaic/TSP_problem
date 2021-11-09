import numpy as np

def calc_solution_objective(solution):

    objective = solution.objective
    solution.vehicle_cost = 100 * len(solution.route_list)

    solution.capacity_violation = 0
    solution.time_violation = 0
    solution.distance_violation = 0
    solution.angle_violation = 0
    solution.node_number_violation = 0
    for route in solution.route_list:
        solution.capacity_violation += 10 ** 6 * route.capacity_violation ** 4
        # asymmetric penalty: since the constraint of feasibility is not a forcible constraint, when there is a violation, little violation has little penalty while much violation has pretty huge penalty.
        solution.time_violation += 10 ** 6 * route.time_violation ** 2
        solution.distance_violation += route.distance_violation
        solution.angle_violation += route.angle_violation
        solution.node_number_violation += 10 ** 4 * route.node_number_violation ** 3

    # intersected_violation = 500 * intersected_node_num

    total_node_silhouette_score = 0
    for rt in solution.route_list:
        node_silhouette_score = 0
        for n in rt.node_list:
            if -0.2 < n.silhouette_score < 0:
                node_silhouette_score += 2.5 * (1 - n.silhouette_score) ** 4
            elif -0.5 < n.silhouette_score <= -0.2:
                node_silhouette_score += 5 * (1 - n.silhouette_score) ** 4
            elif -1 <= n.silhouette_score <= -0.5:
                node_silhouette_score += 10 * (1 - n.silhouette_score) ** 4
            else:
                node_silhouette_score += 1 * (1 - n.silhouette_score) ** 4
        rt.total_node_sil_value = node_silhouette_score
        total_node_silhouette_score += node_silhouette_score

    avg_workload = np.average([route.node_list for route in solution.route_list])
    workload_imbalance = 0
    for route in solution.route_list:
        route.workload_imbalance = (np.abs((route.workload - avg_workload) / avg_workload)) ** 2
        workload_imbalance += route.workload_imbalance
    solution.workload_imbalance = 200 * workload_imbalance

    solution.objective = solution.vehicle_cost + solution.capacity_violation + \
                         solution.time_violation + solution.distance_violation + \
                         solution.angle_violation + solution.node_number_violation + \
                         total_node_silhouette_score + solution.workload_imbalance + \
                         solution.intersected_violation

    return objective


