import os
from ChangeCoordinate import ChangeCoord
from read_problem_data import read_problem_data
from write_problem_data import write_problem_data
from positioning.Gaode_API import ExcuteSingleQuery


# data preprocess
ori_data_path = "./database/2020-11-07上午_data_new.txt"
encoding = "GBK"
problem_data = read_problem_data(ori_data_path, encoding)
# print("read data: ", problem_data.data)

# locationList = [problem_data.data[i]["address"] for i in range(0, len(problem_data.data))]
locationList = []
for i in range(0, len(problem_data.data)):
    if "深圳市" not in problem_data.data[i]["address"]:
        tmp_address = "深圳市" + problem_data.data[i]["address"]
        locationList.append(tmp_address)
    else:
        locationList.append(problem_data.data[i]["address"])
print("locationList", locationList)
print("len(locationList)", len(locationList))

# the maximum number of address translation for GaoDe API is 10
trans_times = int(len(locationList)/10) if len(locationList) % 10 == 0 else int(len(locationList)/10) + 1

# translate address to longitude and latitude
lnglat_list = []
for i in range(0, trans_times):
    tmp_locationList = []
    if i != trans_times - 1:
        tmp_locationList = locationList[10*i: 10*(i+1)]
    else:
        tmp_locationList = locationList[10*i:]

    tmp_lnglatList = ExcuteSingleQuery(locationList=tmp_locationList, currentkey="b36a106aa961abc08e4f9bd60680bd32")
    lnglat_list.extend(tmp_lnglatList)

print("lnglat_list", lnglat_list)
print("len(lnglat_list)", len(lnglat_list))

# translate gcj02 to wgs84
w_lnglatList = []
coord = ChangeCoord()
for i in range(0, len(lnglat_list)):
    lng, lat = coord.gcj02_to_wgs84(lnglat_list[i][0], lnglat_list[i][1])
    w_lnglatList.append([lng, lat])
print("w_lnglatList", w_lnglatList)
print("len(w_lnglatList)", len(w_lnglatList))

for i in range(0, len(problem_data.data)):
    problem_data.data[i]["lnglat"] = w_lnglatList[i]


d = os.path.dirname(__file__)
write_path = d + "/database/2020-11-07上午_data_with_lnglat.txt"
write_problem_data(problem_data, write_path, encoding)


