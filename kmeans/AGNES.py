import math
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def standard(dataset):
    for i in range(len(dataset)):
        dataset[i][0] = dataset[i][0] - 100000000
    return dataset


def euclidean_distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def single_link(x, y):
    return min(euclidean_distance(i, j) for i in x for j in y)


def average_link(x, y):
    pass
    return sum(euclidean_distance(i, j) for i in x for j in y) / (len(x) * len(y))


def complete_link(x, y):
    pass
    return max(euclidean_distance(i, j) for i in x for j in y)


def centroid_link(x, y):
    pass
    return euclidean_distance(np.mean(x), np.mean(y))


def find(mat):
    min = 0x3fffffff
    x = 0
    y = 0
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if i != j and mat[i][j] < min:
                min = mat[i][j]
                x = i
                y = j
    return x, y, min


def AGNES(dataset, method, k):
    cluster_set = []
    dis_matrix = []
    for node in dataset:
        cluster_set.append([node])
    # print('original cluster_set:\n', cluster_set)
    for x in cluster_set:
        dis_list = []
        for y in cluster_set:
            dis_list.append(method(x, y))
        dis_matrix.append(dis_list)
    # for i in range(len(dis_matrix)):
    # print(dis_matrix[i])
    l = len(dataset)

    while l > k:
        index_x, index_y, min_dis = find(dis_matrix)
        # print(index_x, index_y, min_dis)
        cluster_set[index_x].extend(cluster_set[index_y])
        # for i in range(len(cluster_set)):
        #     print(cluster_set[i])
        del cluster_set[index_y]
        dis_matrix = []

        for x in cluster_set:
            dis_list = []
            for y in cluster_set:
                dis_list.append(method(x, y))
            dis_matrix.append(dis_list)
        l -= 1

    return cluster_set


def calculate_distance(a, b):
    return np.linalg.norm(a - b)


def calculate_a(i, cluster, data):
    a = 0
    if len(cluster) - 1 == 0:
        return 0
    else:
        for j in range(len(cluster)):
            if not np.array_equal(data[i], data[j]):
                a += calculate_distance(data[i], data[j])
    return a / (len(cluster) - 1)


def calculate_b(i, cluster_set, data, labels):
    b_values = []
    for other_cluster in cluster_set:
        if not np.array_equal(other_cluster, cluster_set[labels[i]]):
            b = 0
            for j in range(len(other_cluster)):
                b += calculate_distance(data[i], data[j])
            b_values.append(b / len(other_cluster))
    return min(b_values) if len(b_values) > 0 else 0


def silhouette_coefficient(data, labels, cluster_set):
    s_values = []
    for i in range(len(data)):
        a_i = calculate_a(i, cluster_set[labels[i]], data)
        b_i = calculate_b(i, cluster_set, data, labels)
        if a_i == 0 and b_i == 0:
            s_i = 0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)
        s_values.append(s_i)

    silhouette_avg = np.mean(s_values)
    return silhouette_avg


def davies_bouldin(cluster_set):
    k = len(cluster_set)
    cluster_centers = [np.mean(cluster, axis=0) for cluster in cluster_set]

    # 计算簇内平均距离
    avg_in_cluster_list = []
    for i in range(k):
        in_cluster_dis = [calculate_distance(point, cluster_centers[i]) for point in cluster_set[i]]
        avg_int_cluster_dis = np.mean(in_cluster_dis)
        avg_in_cluster_list.append(avg_int_cluster_dis)

    # 计算簇间距离
    in_cluster_dis = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            in_cluster_dis[i][j] = calculate_distance(cluster_centers[i], cluster_centers[j])
            in_cluster_dis[j][i] = in_cluster_dis[i][j]

    # 计算 DBI
    db_index = 0
    for i in range(k):
        max_db_val = 0
        for j in range(k):
            if j != i:
                db_val = (avg_in_cluster_list[i] + avg_in_cluster_list[j]) / in_cluster_dis[i][
                    j]
                if db_val > max_db_val:
                    max_db_val = db_val
        db_index += max_db_val

    db_index /= k

    return db_index


if __name__ == '__main__':
    path = './segmentation data.csv'
    dataset = pd.read_csv(path, index_col='ID')
    dataset = np.array(dataset[:300])
    #dataset = standard(dataset)
    k = 75
    left = 0
    score = []
    DBI = []
    for i in range(k, left, -1):
        cluster_set = AGNES(dataset, average_link, i)
        labels = np.zeros(len(dataset), dtype=int)
        print(i)
        for j, cluster in enumerate(cluster_set):
            for point in cluster:
                labels[point[0] - 1] = j
        #score.append(silhouette_coefficient(dataset, labels, cluster_set))
        DBI.append(davies_bouldin(cluster_set))
    # plt.plot(range(k, left, -1), score, marker='o')
    # plt.xlabel("K")
    # plt.ylabel("silhouette score")
    # plt.show()

    plt.plot(range(k, left, -1), DBI, marker='o')
    plt.xlabel("K")
    plt.ylabel("Davies-Bouldin")
    plt.show()
