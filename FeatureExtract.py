import utils
import network
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import FastICA

def layer_select(test_model):
    flag=0
    for layer in test_model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            flag+=1
            if flag==6:
                _weight, _= layer.get_weights()
                input_channel = layer.input_shape[-1]
                output_channel = layer.filters
                return _weight, input_channel, output_channel

def density_entropy(X):
    K = 3
    N, C, D = X.shape
    x = tf.transpose(X, perm=[1, 0, 2])
    x = tf.reshape(x, [C, N, -1])

    score = []
    for c in range(C):
        nbrs = NearestNeighbors(n_neighbors=K + 1).fit(x[c])
        dms = []
        for i in range(N):
            dm = 0
            dist, ind = nbrs.kneighbors(x[c][i][tf.newaxis, :])
            for j, id in enumerate(ind[0][1:]):
                dm += dist[0][j + 1]

            dms.append(dm)
        dms_sum = tf.reduce_sum(dms)
        en = 0
        for i in range(N):
            en += -dms[i]/dms_sum * tf.math.log(dms[i]/dms_sum) / tf.math.log(tf.constant(2.0, dtype=tf.float64))
        score.append(en)

    return np.array(score)

def ks_compute(X, input_cha, output_cha):
    weight = tf.transpose(X, [2, 3, 0, 1])
    weight = tf.reshape(weight, [output_cha, input_cha, -1])
    ks_weight = np.sum(np.linalg.norm(weight, ord=1, axis=2), 1)
    ks_weight = (ks_weight - np.min(ks_weight)) / (np.max(ks_weight) - np.min(ks_weight))
    return ks_weight

def ke_compute(X, input_cha, output_cha):
    weight = tf.reshape(X, [input_cha, output_cha, -1])
    ke_weight = density_entropy(weight)
    ke_weight = (ke_weight - np.min(ke_weight)) / (np.max(ke_weight) - np.min(ke_weight))
    return ke_weight

def indicator_compute(ks, ke):
    indicator = np.sqrt(ks / (1 + ke))
    indicator = (indicator - np.min(indicator)) / (np.max(indicator) - np.min(indicator))
    return  indicator

def dimension_reduction(data):
    ica = FastICA(n_components=2)
    data_after_reduction = ica.fit_transform(data)

    return data_after_reduction

if __name__ == '__main__':
    utils = utils.GeneralUtils()
    network = network.FCNetwork()
    model = network.vgg16()
    model = network.compile_vgg16(model)
    model.load_weights("XXX")

    weights, input_channels, output_channels = layer_select(model)
    impact_factor = indicator_compute(ks_compute(weights, input_channels, output_channels), ke_compute(weights, input_channels, output_channels))
    print(impact_factor)








