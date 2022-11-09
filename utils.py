# 所需的库： pandas,  numpy, folium
# 请注意所读取csv文件的路径是否和代码中相符
# 不保证代码能够在您的机器上正常运行，所有内容仅供参考

# 帮助1： 数据读取子程序
import pandas as pd
cases_data = pd.read_csv("../R3-case-1.csv", encoding = "utf-8", skiprows=[0])
distance_data = pd.read_csv("../R1-distance.csv",header=None)
link_data = pd.read_csv("../R2-link.csv")
cases =[]
for i in range(cases_data.shape[0]):
    d = dict(cases_data.iloc[i])
    cases.append(d)
#print("Cases列表：", cases)
print("Cases字段：", cases_data.keys())
print("访问一个case:", cases[0])
print("访问一个case中某个值:", cases[0]['纬度.2'])
candidates = eval(cases[0]['经纬度'])
print("访问一个case中起点1所有候选点的经纬度:", candidates)

# 数据建图
G_edges=[]
G_nodes_dic={}
for i in range(link_data.shape[0]):
    dic = dict(link_data.iloc[i])
    G_nodes_dic[dic['Node_Start']]=(dic['Longitude_Start'],dic['Latitude_Start'])
    G_edges.append((int(dic['Node_Start']), int(dic['Node_End']),dic['Length']))
print("节点数量：", len(list(G_nodes_dic.keys())))
print("边数量", len(G_edges))
print("输出节点3的经纬度：",G_nodes_dic[3])
print("输出第0条边的起点、终点、长度：",G_edges[0])

#-----------------------------------
#帮助2： 距离计算
import numpy as np
cnt =0 # 查表次数
distance_dic = np.zeros(distance_data.shape)
def d(x,y): #这是两点间最短距离查询，查表次数记录在cnt变量中，重复查询同样的两点间距离不记次数
    global cnt, distance_dic,distance_data
    if x==y:
        return 0
    if distance_dic[x][y]<1e-5:  #没存过
        distance_dic[x][y] = distance_data[x][y]
        cnt+=1
    return distance_dic[x][y]
#序列距离函数
#Input:
#S为List存储的序列，序列中每个元素为一个位置p，每个位置p应该包含[经度，纬度，邻近Link，所占比例]四种信息
#例: S=[pc,po1,po2,pd1,pd2]  其中 po2=[104.092231,30.687216,946,0.5]
#Output: 输入S序列对应的路径距离
def compute_distance(S):
    distance=0
    #每一轮计算p_i到p_{i+1}之间的距离
    for i in range(len(S)-1):
        #Link_O 为出发点对应的边的编号
        Link_O=S[i][2]
        #Ratio_O 为出发点在其对应边上的比例
        Ratio_O=S[i][3]
        #Link_D 为目的地对应的边的编号
        Link_D=S[i+1][2]
        #Ratio_D 为目的地在其对应边上的比例
        Ratio_D=S[i+1][3]
        #Node_1为出发点对应边的终点
        Node_1=link_data[Link_O-1][4]
        #Node_2为目的地对应边的起点
        Node_2=link_data[Link_D-1][1]
        distance+=d(Node_1,Node_2)
        #加上出发点到出发点所在边终点的距离
        distance+=link_data[Link_O-1][7]*(1-Ratio_O)
        #加上目的地所在边起点到目的地的距离
        distance+=link_data[Link_D-1][7]*Ratio_D
    return distance

"""可以在这一段实现你的计算程序








print("该程序计算过程中查询两点间最短路的次数为:",cnt)
"""

#-----------------------------------
#帮助3：地图可视化
import folium
from folium import plugins,PolyLine
# 以第一个case为例，可视化S=[pc,po1,po2,pd1,pd2]的这条路线
case= cases[0]
pc= (case['经度.6'],case['纬度.6'])
po1= (case['经度'],case['纬度'])
po2= (case['经度.1'],case['纬度.1'])
pd1= (case['经度.2'],case['纬度.2'])
pd2= (case['经度.3'],case['纬度.3'])

map1 = folium.Map(   #高德底图
     location=[30.66,104.11],
     zoom_start=13,
     control_scale = True,
     tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
     attr='&copy; <a href="http://ditu.amap.com/">高德地图</a>'
)

#folium标记函数中第一个参数是位置，先纬度，后经度
folium.Marker([pc[1], pc[0]], popup='<i>车点</i>', icon=folium.Icon(icon='home', color='orange')).add_to(map1)
folium.Marker([po1[1], po1[0]], popup='<i>起点1</i>', icon=folium.Icon(icon='cloud', color='blue')).add_to(map1)
folium.Marker([po2[1], po2[0]], popup='<i>起点2</i>', icon=folium.Icon(icon='cloud', color='red')).add_to(map1)
folium.Marker([pd1[1], pd1[0]], popup='<i>终点1</i>', icon=folium.Icon(icon='ok-sign', color='blue')).add_to(map1)
folium.Marker([pd2[1], pd2[0]], popup='<i>终点1</i>', icon=folium.Icon(icon='ok-sign', color='red')).add_to(map1)

PolyLine([(pc[1],pc[0])  ,(po1[1],po1[0]) ], weight=5, color='blue', opacity=0.8).add_to(map1)
PolyLine([(po1[1],po1[0])  ,(po2[1],po2[0]) ], weight=5, color='blue', opacity=0.8).add_to(map1)
PolyLine([(po2[1],po2[0])  ,(pd1[1],pd1[0]) ], weight=5, color='blue', opacity=0.8).add_to(map1)
PolyLine([(pd1[1],pd1[0])  ,(pd2[1],pd2[0]) ], weight=5, color='blue', opacity=0.8).add_to(map1)
map1.save("Map_visualization.html")  #生成网页
