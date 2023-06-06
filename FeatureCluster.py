import hdbscan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from FeatureExtract import dimension_reduction

# from mmd_critic import MMD_Critic
sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}

def cluster_hdbscan(data_feature):
    # print(data_feature)
    hdbscan_result = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    hdbscan_result.fit(data_feature)
    return hdbscan_result

def cluster_minimum_tree(cluster_result):
    cluster_result.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)
    plt.show()

def cluster_condensed_tree(cluster_result):
    cluster_result.condensed_tree_.plot()
    plt.show()

def cluster_condensed_selection(cluster_result):
    cluster_result.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
    plt.show()

def cluster_colors(cluster_result, data_feature):
    n_clusters = len(set(cluster_result.labels_))
    palette = sns.color_palette(n_colors=n_clusters)
    cluster_colors = [sns.desaturate(palette[col], sat) if col >= 0 else (0.5, 0.5, 0.5) for col, sat in zip(cluster_result.labels_, cluster_result.probabilities_)]
    x,y = [],[]
    for i in data_feature:
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x, y, c= cluster_colors, **plot_kwds)
    plt.show()

def cluster_center(cluster_result, data_feature):
    center_indices = []

    for i in range(np.max(cluster_result.labels_) + 1):
        mask = (cluster_result.labels_ == i)
        if np.any(mask):
            cluster_data = data_feature[mask]
            kmeans = KMeans(n_clusters=1)
            kmeans.fit(cluster_data)
            center = kmeans.cluster_centers_

            dists = pairwise_distances(cluster_data, center, metric='euclidean').squeeze()
            sorted_indices = np.argsort(dists)

            num_points = int(np.ceil(cluster_data.shape[0] * k))
            center_indices.extend(np.where(mask)[0][sorted_indices[:num_points]])

    print("center data--------------------------", len(center_indices))
    for j in range(len(center_indices)):
        print(j, center_indices[j], data_feature[j])

    return center_indices


def cluster_outlier(cluster_result, data_feature):
    outlier_mask = (cluster_result.labels_ == -1)
    outlier_indices = np.where(outlier_mask)[0]
    outliers = data_feature[outlier_mask]

    print("outlier_data--------------------------",len(outlier_indices))
    for j in range(len(outlier_indices)):
        print(j, outlier_indices[j], data_feature[j])

    return outlier_indices


if __name__ == '__main__':

    test_data,_ = data.make_blobs(n_samples=1000, n_features=6, centers=6)
    test_data = dimension_reduction(test_data)
    after_cluster = cluster_hdbscan(test_data)

    center_index = cluster_center(after_cluster,test_data)
    outlier_index = cluster_outlier(after_cluster, test_data)






