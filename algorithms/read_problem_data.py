import json
import os
from models.problem_data import ProblemData
from write_problem_data import write_problem_data
from check_problem_data import check_problem_data


def read_problem_data(path, encoding):
    problem_data = ProblemData()

    with open(path, encoding=encoding) as f:
        load_dict = json.load(f)
        # print("load_dict", load_dict)

        try:
            # check whether the .txt has all the problem data or not
            # read some parameters
            problem_data.maxDurationList = load_dict["config"]["maxDurationList"]
            problem_data.angles = load_dict["config"]["angles"]
            problem_data.maxNodeNumbers = load_dict["config"]["maxNodeNumbers"]
            problem_data.nonImproveList = load_dict["config"]["nonImproveList"]
            problem_data.operatorDurationList = load_dict["config"]["operatorDurationList"]

            data = load_dict["data"]
            chosen_data = []
            # read each instance
            for i in range(0, len(data)):
                d = {}

                d["from"] = data[i]["from"]
                d["volume"] = data[i]["volume"]
                d["weight"] = data[i]["weight"]
                d["carrierId"] = data[i]["carrierId"]
                d["address"] = data[i]["address"]
                d["transportTime"] = data[i]["transportTime"]
                if "lnglat" in data[i]:
                    d["lnglat"] = data[i]["lnglat"]

                chosen_data.append(d)

            problem_data.data = chosen_data
            # return problem_data
        except KeyError:
            print("Can not get the right problem data.")

        if check_problem_data(problem_data):
            return problem_data
        else:
            print("incomplete problem data")
            return None


# if __name__ == '__main__':
# ori_data_path = "./database/2020-11-07上午_data.txt"
# encoding = "GBK"
# problem_data = read_problem_data(ori_data_path, encoding)
# print("read data: ", problem_data.data)
#
# d = os.path.dirname(__file__)
#
# write_path = d + "/database/2020-11-07上午_data_new.txt"
# write_problem_data(problem_data, write_path, encoding)
