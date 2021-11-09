import json
from check_problem_data import check_problem_data


def write_problem_data(problem_data, write_path, encoding):
    if check_problem_data(problem_data):
        write_dict = {}
        config = {}

        config["maxDurationList"] = problem_data.maxDurationList
        config["angles"] = problem_data.angles
        config["maxNodeNumbers"] = problem_data.maxNodeNumbers
        config["nonImproveList"] = problem_data.nonImproveList
        config["operatorDurationList"] = problem_data.operatorDurationList

        write_dict["config"] = config
        write_dict["data"] = problem_data.data

        json_str = json.dumps(write_dict, ensure_ascii=False, indent=4)  # 缩进4字符
        with open(write_path, 'w', encoding=encoding) as json_file:
            json_file.write(json_str)
    else:
        print("incomplete problem data")