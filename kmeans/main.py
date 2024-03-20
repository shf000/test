import sys

import pandas as pd
from matplotlib import pyplot
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score

if __name__ == '__main__':
    path = './segmentation data.csv'
    data = pd.read_csv(path, index_col='ID')
    data = data.drop(data.index[0])
    data = data.reset_index(drop=True)
    ss = StandardScaler()
    data = ss.fit_transform(data)
    random_seed = 0
    left = 2
    k = 1998
    score = []
    SSE = []
    CH = []
    DBI = []
    for i in range(left, k+1):
        kmeans = KMeans(n_clusters=i, n_init='auto')
        kmeans.fit(data)
        pred = kmeans.predict(data)
        if i % 50 == 0:
            print(i)
        #score.append(silhouette_score(data, kmeans.labels_))
        #SSE.append(kmeans.inertia_)
        CH.append(calinski_harabasz_score(data, kmeans.labels_))
        # DBI.append(davies_bouldin_score(data, kmeans.labels_))
    # plt.plot(range(left, k+1), score, marker='o')
    # plt.xlabel("K")
    # plt.ylabel("silhouette score")
    # plt.show()

    # plt.plot(range(left, k+1), SSE, marker='o')
    # plt.xlabel("K")
    # plt.ylabel("SSE")
    # plt.show()

    plt.plot(range(left, k+1), CH, marker='o')
    plt.xlabel("K")
    plt.ylabel("Calinski_Harabasz")
    plt.show()

    # plt.plot(range(left, k+1), DBI, marker='o')
    # plt.xlabel("K")
    # plt.ylabel("Davies-Bouldin")
    # plt.show()