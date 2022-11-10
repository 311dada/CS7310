import argparse
import pandas as pd
import logging
import numpy as np
import folium
from folium import PolyLine
from typing import List
import itertools
from geopy.distance import geodesic

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

cnt = 0
distance_dic = None

name2field = {
    "c": ("经度.6", "纬度.6", "所在link", "ratio.6"),
    "o1": ("经度", "纬度", "邻近link", "ratio"),
    "d1": ("经度.1", "纬度.1", "邻近link.1", "ratio.1"),
    "o2": ("经度.2", "纬度.2", "邻近link.2", "ratio.2"),
    "d2": ("经度.3", "纬度.3", "邻近link.3", "ratio.3"),
    "o3": ("经度.4", "纬度.4", "邻近link.4", "ratio.4"),
    "d3": ("经度.5", "纬度.5", "邻近link.5", "ratio.5"),
}


def parse():
    parser = argparse.ArgumentParser("==== CS7310 Group Project Parser ====")

    parser.add_argument(
        "--case_file", type=str, default="../R3-case-11.csv", help="case csv file"
    )
    parser.add_argument(
        "--distance_file",
        type=str,
        default="../R1-distance.csv",
        help="distance csv file",
    )
    parser.add_argument(
        "--link_file", type=str, default="../R2-link.csv", help="link csv file"
    )

    return parser.parse_args()


def load_data(args):
    cases_data = pd.read_csv(args.case_file, encoding="utf-8", skiprows=[0])
    distance_data = pd.read_csv(args.distance_file, header=None)
    link_data = pd.read_csv(args.link_file)

    cases = []
    for i in range(cases_data.shape[0]):
        d = dict(cases_data.iloc[i])
        cases.append(d)

    case_field = "\n".join(cases_data.keys())
    logger.info(f"Cases fields:\n {case_field}")

    return distance_data, link_data, cases


def build_graph(link_data):
    G_edges = []
    G_nodes_dic = {}
    for i in range(link_data.shape[0]):
        dic = dict(link_data.iloc[i])
        G_nodes_dic[dic["Node_Start"]] = (dic["Longitude_Start"], dic["Latitude_Start"])
        G_edges.append((int(dic["Node_Start"]), int(dic["Node_End"]), dic["Length"]))

    logger.info(f"Node number: {len(G_nodes_dic)}")
    logger.info(f"Edge number: {len(G_edges)}")

    return G_edges, G_nodes_dic


def d(x, y, distance_data):
    global cnt, distance_dic
    if x == y:
        return 0
    if distance_dic[x][y] < 1e-5:  # 没存过
        distance_dic[x][y] = distance_data[x][y]
        cnt += 1
    return distance_dic[x][y]


# 序列距离函数
# Input:
# S为List存储的序列，序列中每个元素为一个位置p，每个位置p应该包含[经度，纬度，邻近Link，所占比例]四种信息
# 例: S=[pc,po1,po2,pd1,pd2]  其中 po2=[104.092231,30.687216,946,0.5]
# Output: 输入S序列对应的路径距离
def compute_distance(S, link_data, distance_data):
    distance = 0
    # 每一轮计算p_i到p_{i+1}之间的距离
    for i in range(len(S) - 1):
        # Link_O 为出发点对应的边的编号
        Link_O = S[i][2]
        # Ratio_O 为出发点在其对应边上的比例
        Ratio_O = S[i][3]
        # Link_D 为目的地对应的边的编号
        Link_D = S[i + 1][2]
        # Ratio_D 为目的地在其对应边上的比例
        Ratio_D = S[i + 1][3]
        # Node_1为出发点对应边的终点
        Node_1 = int(link_data[Link_O - 1][4])
        # Node_2为目的地对应边的起点
        Node_2 = int(link_data[Link_D - 1][1])
        distance += d(Node_1, Node_2, distance_data)
        # 加上出发点到出发点所在边终点的距离
        distance += link_data[Link_O - 1][7] * (1 - Ratio_O)
        # 加上目的地所在边起点到目的地的距离
        distance += link_data[Link_D - 1][7] * Ratio_D
    return distance


def is_valid_seq(S: List[str]):
    # beginning with car
    if S[0] != "c":
        return False

    # oi must be earlier than di
    role2idx = dict()
    for i in range(1, len(S)):
        role2idx[S[i]] = i

    # no independent passenger
    left = 0
    for i in range(1, len(S)):
        if S[i][0] == "o":
            left += 1
        else:
            left -= 1

        if not left and i < len(S) - 1:
            return False

    for i in range(1, (len(S) + 1) // 2):
        if role2idx[f"o{i}"] > role2idx[f"d{i}"]:
            return False

    return True


def generate_valid_seqs(p_num):
    base = ["c"]
    for i in range(1, p_num + 1):
        base.extend([f"o{i}", f"d{i}"])

    valid_seqs = []
    for seq in itertools.permutations(base):
        if is_valid_seq(seq):
            valid_seqs.append(seq)

    return valid_seqs


def get_accumu_counts(seqs):
    counted = set()
    counts = []
    for seq in seqs:
        counted |= set([(seq[i], seq[i + 1]) for i in range(len(seq) - 1)])
        counts.append(len(counted))

    return counts


def seq2tex(seq):
    return "".join(
        map(lambda x: r"p_{c}" if x == "c" else "p_{" + x[0] + "}^{" + x[-1] + "}", seq)
    )


def traverse(cases, seqs_list, link_data, distance_data):
    optimal = []
    optimal_seqs = []
    table = ""
    for i, case in enumerate(cases):
        optimal_seq = None
        minm_dist = float("inf")
        # seqs = two_passengers_seqs if i < 5 else three_passengers_seqs
        seqs = seqs_list[i]
        for seq in seqs:
            positions = []
            for pos in seq:
                positions.append([case[field] for field in name2field[pos]])

            dist = compute_distance(positions, link_data, distance_data)

            if dist < minm_dist:
                optimal_seq = seq
                minm_dist = dist

        line = "Case "

        if i + 1 < 10:
            line += "0"
        line += str(i + 1) + " & "

        if i < 5:
            line += "双拼 & "
        else:
            line += "三拼 & "

        line += seq2tex(optimal_seq) + " & "
        line += str(minm_dist) + " \\\\\n"

        table += line

        optimal.append(minm_dist)
        optimal_seqs.append(optimal_seq)

    return optimal, table, optimal_seqs


def task1(link_data, distance_data, cases):
    logger.info("Start task 1 !!!")
    # part 1
    def get_table(p_num):
        seqs = generate_valid_seqs(p_num)
        counts = get_accumu_counts(seqs)

        display_str = f"table for {p_num} passengers of task 1.1\n\n"
        for i, seq in enumerate(seqs):
            display_str += f"{i + 1} & {seq2tex(seq)} & {counts[i]} \\\\\n"

        logger.info(display_str)

        return seqs

    two_passengers_seqs = get_table(2)
    three_passengers_seqs = get_table(3)

    # part 2
    # two passengers
    seqs_list = [
        two_passengers_seqs if i < 5 else three_passengers_seqs for i in range(10)
    ]

    table = "table for task 1.2\n\n"

    optimal, table_content, _ = traverse(cases, seqs_list, link_data, distance_data)

    table += table_content

    logger.info(table)
    return two_passengers_seqs, three_passengers_seqs, optimal


def filter_by_spherical_distance(cases, seqs_list, threshold=100):
    filtered_seqs_list = []
    for i, case in enumerate(cases):
        seqs = seqs_list[i]

        seqs_with_spherical_distance = []
        for seq in seqs:
            distance = 0
            for i in range(len(seq) - 1):
                lng1, lat1 = case[name2field[seq[i]][0]], case[name2field[seq[i]][1]]
                lng2, lat2 = (
                    case[name2field[seq[i + 1]][0]],
                    case[name2field[seq[i + 1]][1]],
                )

                distance += geodesic((lat1, lng1), (lat2, lng2)).m

            seqs_with_spherical_distance.append((seq, distance))

        seqs_with_spherical_distance = sorted(
            seqs_with_spherical_distance, key=lambda x: x[-1]
        )
        base = seqs_with_spherical_distance[0][-1]

        filtered_seqs = [seqs_with_spherical_distance[0][0]]

        for i in range(1, len(seqs_with_spherical_distance)):
            if seqs_with_spherical_distance[i][-1] - base <= threshold:
                filtered_seqs.append(seqs_with_spherical_distance[i][0])
            else:
                break

        filtered_seqs_list.append(filtered_seqs)

    return filtered_seqs_list


# TODO
def task2(
    two_passengers_seqs, three_passengers_seqs, optimal, cases, link_data, distance_data
):
    logger.info("Start task 2 !!!")

    # method 1: by spherical distance
    seqs_list = [
        two_passengers_seqs if i < 5 else three_passengers_seqs for i in range(10)
    ]
    filtered_seqs_list = filter_by_spherical_distance(cases, seqs_list, 1727.5)

    table = "table for task 2\n\n"

    fake_optimal, table_content, fake_optimal_seqs = traverse(
        cases, filtered_seqs_list, link_data, distance_data
    )

    counts = [get_accumu_counts(seqs)[-1] for seqs in filtered_seqs_list]

    for i in range(10):
        line = "Case "

        if i + 1 < 10:
            line += "0"
        line += str(i + 1) + " & "

        if i < 5:
            line += "双拼 & "
        else:
            line += "三拼 & "
        
        line += seq2tex(fake_optimal_seqs[i]) + " & "
        line += str(fake_optimal[i]) + " & "
        line += str(fake_optimal[i] / optimal[i] * 100) + " & "
        line += str(counts[i]) + "\n"

        table += line

    logger.info(table)


# TODO
def task3():
    logger.info("Start task 3 !!!")


# TODO
def task4():
    logger.info("Start task 4 !!!")


def visualize(cases):
    case = cases[0]
    pc = (case["经度.6"], case["纬度.6"])
    po1 = (case["经度"], case["纬度"])
    po2 = (case["经度.1"], case["纬度.1"])
    pd1 = (case["经度.2"], case["纬度.2"])
    pd2 = (case["经度.3"], case["纬度.3"])

    map1 = folium.Map(  # 高德底图
        location=[30.66, 104.11],
        zoom_start=13,
        control_scale=True,
        tiles="http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
        attr='&copy; <a href="http://ditu.amap.com/">高德地图</a>',
    )

    # folium标记函数中第一个参数是位置，先纬度，后经度
    folium.Marker(
        [pc[1], pc[0]], popup="<i>车点</i>", icon=folium.Icon(icon="home", color="orange")
    ).add_to(map1)
    folium.Marker(
        [po1[1], po1[0]],
        popup="<i>起点1</i>",
        icon=folium.Icon(icon="cloud", color="blue"),
    ).add_to(map1)
    folium.Marker(
        [po2[1], po2[0]],
        popup="<i>起点2</i>",
        icon=folium.Icon(icon="cloud", color="red"),
    ).add_to(map1)
    folium.Marker(
        [pd1[1], pd1[0]],
        popup="<i>终点1</i>",
        icon=folium.Icon(icon="ok-sign", color="blue"),
    ).add_to(map1)
    folium.Marker(
        [pd2[1], pd2[0]],
        popup="<i>终点1</i>",
        icon=folium.Icon(icon="ok-sign", color="red"),
    ).add_to(map1)

    PolyLine(
        [(pc[1], pc[0]), (po1[1], po1[0])], weight=5, color="blue", opacity=0.8
    ).add_to(map1)
    PolyLine(
        [(po1[1], po1[0]), (po2[1], po2[0])], weight=5, color="blue", opacity=0.8
    ).add_to(map1)
    PolyLine(
        [(po2[1], po2[0]), (pd1[1], pd1[0])], weight=5, color="blue", opacity=0.8
    ).add_to(map1)
    PolyLine(
        [(pd1[1], pd1[0]), (pd2[1], pd2[0])], weight=5, color="blue", opacity=0.8
    ).add_to(map1)
    map1.save("Map_visualization.html")  # 生成网页


if __name__ == "__main__":
    args = parse()
    distance_data, link_data, cases = load_data(args)
    # G_edges, G_nodes_dic = build_graph(link_data)

    distance_dic = np.zeros(distance_data.shape)

    two_passengers_seqs, three_passengers_seqs, optimal = task1(
        link_data.values, distance_data, cases
    )

    task2(
        two_passengers_seqs,
        three_passengers_seqs,
        optimal,
        cases,
        link_data.values,
        distance_data,
    )
    task3()
    task4()

    # visualize(cases)
