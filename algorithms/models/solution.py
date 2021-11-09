class Solution(object):
    def __init__(self):
        self.route_list = []
        self.vehicleId_list = []  # if all the vehicles have the same capacity, it doesn't important
        self.separate_nodes = []

        self.relaxed_nodes = []  # for alns since it has destroy and repair method

        self.objective = 1000000000  # a very large value

        self.feasibility = None

        # some evaluation indexes
        self.capacity_violation = 0
        self.time_violation = 0
        self.distance_violation = 0
        self.angle_violation = 0
        self.node_number_violation = 0
        self.intersected_violation = 0
        self.workload_imbalance = 0
        self.silhouette_score = 0
        self.vehicle_cost = 0







