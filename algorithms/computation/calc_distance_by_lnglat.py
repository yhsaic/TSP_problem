import json
from geopy.distance import geodesic
from read_problem_data import read_problem_data

ori_data_path = "../database/2020-11-07上午_data_with_lnglat.txt"
encoding = "GBK"
problem_data = read_problem_data(ori_data_path, encoding)

# computation format: (lat, lng)
start_point = (22.690810, 114.227667)
print(problem_data.data[0])

lnglat_list = [problem_data.data[i]["lnglat"] for i in range(0, len(problem_data.data))]
latlng_list = [start_point] + [(lnglat_list[i][1], lnglat_list[i][0]) for i in range(0, len(lnglat_list))]

# compute the distances of each two points
distance_dict = {}
for i in range(1, len(latlng_list)):
    for j in range(0, i):
        tmp_str1 = str(i) + "_" + str(j)
        tmp_str2 = str(j) + "_" + str(i)
        tmp_distance = geodesic(latlng_list[i], latlng_list[j]).km
        distance_dict[tmp_str1] = tmp_distance
        distance_dict[tmp_str2] = tmp_distance


print(distance_dict)
print(len(distance_dict.keys()))
write_path = "../database/2020-11-07上午_distance.txt"
json_str = json.dumps(distance_dict, ensure_ascii=False, indent=4)  # 缩进4字符
with open(write_path, 'w', encoding=encoding) as json_file:
    json_file.write(json_str)



