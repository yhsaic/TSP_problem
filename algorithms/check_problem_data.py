def check_problem_data(problem_data):
    # check if the parameters are not equal to 0
    if not problem_data.maxDurationList or not problem_data.angles\
            or not problem_data.maxNodeNumbers or not problem_data.nonImproveList\
            or not problem_data.operatorDurationList:
        return False

    # for each instance, check if all the indexes have values
    data = problem_data.data
    for i in range(0, len(data)):
        if data[i]["from"] == "" or not data[i]["volume"]\
                or not data[i]["weight"] or data[i]["carrierId"] == ""\
                or data[i]["address"] == "" or not data[i]["transportTime"]:
            return False

    return True


