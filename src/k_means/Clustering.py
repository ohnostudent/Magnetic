# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
from glob import glob
sys.path.append(os.getcwd() + "/src")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from Visualization.SnapData import SnapData
from params import SNAP_PATH, ML_RESULT_DIR


class ClusteringMethod(SnapData):
    def compress(self, array, LEVEL=10):
        return self._convolute(array, self._ave_carnel(LEVEL), stride = LEVEL)

    def load_regularize(self, item):
        type = item[:-4]
        if type == ".npy":
            im = np.load(item)

        elif type == ".npz":
            print("npz doesnot supported")
            return
        
        else:
            im = self.loadSnapData(item, z=3)
        img_resize = self.compress(im)
        return ((img_resize - min(img_resize.flat)) / max(img_resize.flat)).flat # 正規化

    def PCA(self, X_train):
        #PCA
        N_dim =  100 # 49152(=128×128×3)の列を100列に落とし込む
        pca = PCA(n_components=N_dim, random_state=0)
        X_train_pca = pca.fit_transform(X_train)
        print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))

        return X_train_pca

    def KMeans(self, X_train_pca):
        model = KMeans(n_clusters=4, random_state=1)
        model.fit(X_train_pca)
        cluster = model.labels_#ラベルを返す

        return cluster

    def save_result(self, cluster, path_list, dataset, save=True):
        index = map(os.path.basename, path_list)
        columns = ["cluster"]
        df_clustering_result = pd.DataFrame(cluster, index=index, columns=columns)
        df_clustering_result = df_clustering_result.reset_index()
        df_clustering_result[['parameter', 'param', 'job']] = df_clustering_result["index"].str.split('.', expand=True)
        df_clustering_result = df_clustering_result[['parameter', 'param', 'job', 'cluster']]
        
        if save:
            df_clustering_result.to_csv(ML_RESULT_DIR + f"/clustering/snap{dataset}{self.val_param}.csv", encoding="utf-8")

        return df_clustering_result
        

def doClustering():
    from params import datasets, variable_parameters
    cluster = ClusteringMethod()

    if not os.path.exists(ML_RESULT_DIR + "/clustering"):
        os.makedirs(ML_RESULT_DIR + "/clustering")

    for dataset in datasets:
        for val_param in variable_parameters:
            path_list = glob(SNAP_PATH + f"/snap{dataset}/{val_param}/*/*")
            num_of_data = len(path_list) # リコネクションがない画像の枚数

            temp_data = cluster.compress(cluster.loadSnapData(path_list[0],z=3))
            IMGSHAPE = temp_data.shape

            N_col = IMGSHAPE[0] * IMGSHAPE[1] * 1 # 行列の列数
            X_train = np.zeros((num_of_data, N_col)) # 学習データ格納のためゼロ行列生成
            y_train = np.zeros((num_of_data)) # 学習データに対するラベルを格納するためのゼロ行列生成

            # リコネクションがない画像を行列に読み込む
            for idx, item in enumerate(path_list[:10]):
                X_train[idx, :] = cluster.load_regularize(item)
                y_train[idx] = 0 # リコネクションがないことを表すラベル

            X_train_pca = cluster.PCA(X_train)
            cluster_labels = cluster.KMeans(X_train_pca)
            df_re = cluster.save_result(cluster_labels, path_list, dataset)
            # display(df_re)


if __name__ == "__main__":
    doClustering()
    