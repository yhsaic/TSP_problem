import json
from models.solution import Solution
from check_solution import check_solution


def read_solution(path, encoding, problem_data, para):
    new_solution = Solution()

    with open(path, encoding=encoding) as f:
        load_dict = json.load(f)
        print("load_dict", load_dict)

        try:
            # read some parameters
            new_solution.route_list = load_dict["route_list"]
            new_solution.vehicleId_list = load_dict["vehicleId_list"]
            new_solution.objective = load_dict["objective"]
            new_solution.feasibility = load_dict["feasibility"]

            new_solution.capacity_violation = load_dict["capacity_violation"]
            new_solution.time_violation = load_dict["time_violation"]
            new_solution.distance_violation = load_dict["distance_violation"]
            new_solution.angle_violation = load_dict["angle_violation"]
            new_solution.node_number_violation = load_dict["node_number_violation"]
            new_solution.intersected_violation = load_dict["intersected_violation"]
            new_solution.workload_imbalance = load_dict["workload_imbalance"]
            new_solution.silhouette_score = load_dict["silhouette_score"]
            new_solution.vehicle_cost = load_dict["vehicle_cost"]

        except KeyError:
            print("Can not get the right solution data.")

        if check_solution(problem_data, new_solution, para):
            return new_solution
        else:
            print("invalid solution")
            return None



