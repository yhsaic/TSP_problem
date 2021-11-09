class Route(object):
    def __init__(self):
        self.node_list = []
        self.vehicle_ID = None

        self.volume = 0
        self.weight = 0

        # some evaluation indexes
        self.capacity_violation = 0
        self.time = 0
        self.distance = 0
        self.angle_violation = 0
        self.node_number_violation = 0
        self.workload = 0
        self.total_node_sil_value = 0
