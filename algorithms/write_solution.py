import json
from check_solution import check_solution


def write_solution(solution, problem_data, write_path, encoding, para):
    write_dict = {}

    new_route_list = []
    for rt in solution.route_list:
        new_route_list.append([node.id for node in rt.node_list])

    write_dict["route_list"] = new_route_list
    write_dict["vehicleId_list"] = solution.vehicleId_list
    write_dict["objective"] = solution.objective
    write_dict["feasibility"] = solution.feasibility

    write_dict["capacity_violation"] = solution.capacity_violation
    write_dict["time_violation"] = solution.time_violation
    write_dict["distance_violation"] = solution.distance_violation
    write_dict["angle_violation"] = solution.angle_violation
    write_dict["node_number_violation"] = solution.node_number_violation
    write_dict["intersected_violation"] = solution.intersected_violation
    write_dict["workload_imbalance"] = solution.workload_imbalance
    write_dict["silhouette_score"] = solution.silhouette_score
    write_dict["vehicle_cost"] = solution.vehicle_cost

    if check_solution(problem_data, solution, para):
        json_str = json.dumps(write_dict, ensure_ascii=False, indent=4)  # 缩进4字符
        with open(write_path, 'w', encoding=encoding) as json_file:
            json_file.write(json_str)
    else:
        print("invalid solution")