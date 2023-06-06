import os

import numpy as np

import utils
import network
from FeatureExtract import layer_select, indicator_compute, ks_compute, ke_compute

utils = utils.GeneralUtils()
network = network.FCNetwork()

# model = network.Lenet1()
# model =network.vgg16()
model = network.Cifarmodel()
f = open('XXX', 'r+', encoding="utf-8")
#计算变异体特征
def mutants_feature_compute(filenames_list):
    i = 0
    for filename in filenames_list:
        print(i, filename)
        filename = 'XXX'+filename
        model.load_weights(filename)#tensorflow2
        weights, input_channels, output_channels = layer_select(model)
        impact_factor = indicator_compute(ks_compute(weights, input_channels, output_channels), ke_compute(weights, input_channels, output_channels))
        print(impact_factor)
        i = i + 1
        f.write(str(impact_factor))
        f.write("\n")

if __name__ == '__main__':
    # 读取h5文件路径
    filePath = 'XXX'
    filenames = os.listdir(filePath)  # 返回指定的文件夹下包含的文件或文件夹名字的列表，这个列表按字母顺序排序。
    for fi in filenames:
        print(fi)
    feature_data = mutants_feature_compute(filenames)
    mutants_feature_to_excel(feature_data, 'XXX.xlsx', 'XXX')

    data = read_excel_to_list('XXX.xlsx', 'XXX')
    data = np.array(data)
    data = dimension_reduction(data)
    after_cluster = cluster_hdbscan(data)
    cluster_colors(after_cluster,data)
    cluster_minimum_tree(after_cluster)
    cluster_condensed_tree(after_cluster)
    cluster_condensed_selection(after_cluster)

    center_index = cluster_center(after_cluster,data)
    mutants_number_to_excel(center_index, 'XXX.xlsx', 'XXX')
    outlier_index = cluster_outlier(after_cluster, data)
    mutants_number_to_excel(outlier_index, 'XXX.xlsx', 'XXX')









