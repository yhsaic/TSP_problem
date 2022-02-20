# get the longitude and latitude of specific address
import requests


# 执行一次高德地图地理编码的查询
# 输入值：locationList -> 地址的序列,currentKey -> 当前使用的Key
# 返回值：resultList -> 查询成功，返回结果坐标的序列
#        -1 -> 执行当前查询时Key的配额用完了
#        -2 -> 执行当前查询出错
# reference: https://zhuanlan.zhihu.com/p/102276721
def ExcuteSingleQuery(locationList, currentkey):
    # 1-将locationList中的地址连接成高德地图API能够识别的样子
    locationString = ""     # 当前locationList组成的string
    for location in locationList:
        locationString += location + '|'
    # 2-地理编码查询需要的Url
    output = 'json'    # 查询返回的形式
    batch = 'true'     # 是否支持多个查询
    base = 'https://restapi.amap.com/v3/geocode/geo?'    # 地理编码查询Url的头
    currentUrl = base + "output=" + output + "&batch=" + batch + "&address=" + locationString + "&key=" + currentkey
    # 3-提交请求
    response = requests.get(currentUrl)    # 提交请求
    answer = response.json()   # 接收返回
    # 4-解析Json的内容
    resultList = []    # 用来存放地理编码结果的空序列
    if answer['status'] == '1' and answer['info'] == 'OK':
        # 4.1-请求和返回都成功，则进行解析
        tmpList = answer['geocodes']    # 获取所有结果坐标点
        for i in range(0, len(tmpList)):
            try:
                # 解析','分隔的经纬度
                coordString = tmpList[i]['location']
                coordList = coordString.split(',')
                # 放入结果序列
                resultList.append([float(coordList[0]), float(coordList[1])])
            except:
                # 如果发生错误则存入None
                resultList.append(None)
        return resultList
    elif answer['info'] == 'DAILY_QUERY_OVER_LIMIT':
        # 4.2-当前账号的余额用完了,返回-1
        return -1
    else:
        # 4.3-如果发生其他错误则返回-2
        return -2


# # 创建测试地址数据集
# locationList = [
#     "四川省成都市崇州市崇阳街道晋康北路313号",
#     "四川省成都市成都高新区盛华北路107号1楼",
#     "四川省成都市四川省成都市金堂县赵镇迎宾大道一段8号",
#     "四川省成都市崇州市三江镇崇新村5组",
#     "四川省成都市大邑县鹤鸣乡联和村二十组",
#     "四川省成都市成都市蒲江县复兴乡庙峰村3组52号",
#     "四川省成都市成都高新区新乐路125号1楼",
#     "四川省成都市成都市青羊区锦里西路127号1层3号",
#     "四川省成都市成都高新区天府一街616号8栋附203号",
#     "四川省成都市四川省成都市成华区东三环路二段宝耳路2号1号办公楼3楼1号",
# ]
#
# # 进行地理编码
# print(ExcuteSingleQuery(locationList=locationList, currentkey="b36a106aa961abc08e4f9bd60680bd32"))
