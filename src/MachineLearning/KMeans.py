# -*- coding: utf-8 -*-

import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/src")
from config.params import ML_RESULT_DIR, SNAP_PATH
from Visualization.SnapData import SnapData


class ClusteringMethod(SnapData):
    """

    Args:

    Returns:

    """

    def compress(self, array, level: int = 10) -> np.ndarray:
        """
        畳み込みを行う関数

        Args:
            array() : 畳み込みを行うデータ
            level (int) : stride

        Returns:
            None

        """
        return self._convolute(array, self._ave_kernel(level), stride=level)

    def load_regularize(self, path: str) -> np.ndarray:
        """
        データのロードを行う関数

        Args:
            path (str) : ファイルパス

        Returns:

        """
        type = path[:-4]
        if type == ".npy":
            im = np.load(path)

        elif type == ".npz":
            print("npz does not supported")
            return

        else:  # バイナリの読み込み
            im = self.loadSnapData(path, z=3)

        img_resize = self.compress(im)
        return ((img_resize - min(img_resize.flat)) / max(img_resize.flat)).flat  # 正規化

    def PCA(self, X_train) -> np.ndarray:
        """PCA
        PCA を行う関数

        Args:
            X_train () : 学習用データ

        Returns:
            ndarray :

        """

        # PCA
        N_dim = 100  # 49152(=128×128×3)の列を100列に落とし込む
        pca = PCA(n_components=N_dim, random_state=0)
        X_train_pca = pca.fit_transform(X_train)
        print("累積寄与率: {0}".format(sum(pca.explained_variance_ratio_)))

        return X_train_pca

    def KMeans(self, X_train_pca) -> np.ndarray:
        """
        KMeans を行う関数

        Args:

        Returns:
            ndarray :

        """
        model = KMeans(n_clusters=4, random_state=1)
        model.fit(X_train_pca)
        cluster = model.labels_  # ラベルを返す

        return cluster

    def save_result(self, cluster_labels, path_list: list, dataset: int, save=True) -> pd.DataFrame:
        """
        KMeans の結果を保存する関数

        Args:
            cluster_labels (list) : ラベリング後の配列
            path_list (list) : クラスタリングを行ったファイルのパスの配列
            dataset (int) : 用いたデータセット
            save (bool) : 保存を行うかどうかの変数

        Returns:
            pd.DataFrame

        """
        # pd.DataFrame に変換
        index = map(os.path.basename, path_list)
        columns = ["cluster"]
        df_clustering_result = pd.DataFrame(cluster_labels, index=index, columns=columns)
        df_clustering_result = df_clustering_result.reset_index()
        df_clustering_result[["parameter", "param", "job"]] = df_clustering_result["index"].str.split(".", expand=True)
        df_clustering_result = df_clustering_result[["parameter", "param", "job", "cluster"]]

        # 保存
        if save:
            df_clustering_result.to_csv(ML_RESULT_DIR + f"/clustering/snap{dataset}{self.val_param}.csv", encoding="utf-8")

        return df_clustering_result


def doClustering() -> None:
    from config.params import datasets, variable_parameters
    from config.SetLogger import logger_conf

    logger = logger_conf()
    cluster = ClusteringMethod()

    if not os.path.exists(ML_RESULT_DIR + "/clustering"):
        os.makedirs(ML_RESULT_DIR + "/clustering")

    for dataset in datasets:
        logger.debug("START", extra={"addinfo": f"snap{dataset}"})
        for val_param in variable_parameters:
            logger.debug("START", extra={"addinfo": f"{val_param}"})

            # パスの取得
            path_list = glob(SNAP_PATH + f"/snap{dataset}/{val_param}/*/*")

            num_of_data = len(path_list)  # リコネクションがない画像の枚数
            temp_data = cluster.compress(cluster.loadSnapData(path_list[0], z=3))
            IMG_SHAPE = temp_data.shape  # 画像サイズ

            # 行列の列数
            N_col = IMG_SHAPE[0] * IMG_SHAPE[1] * 1
            # 学習データ格納のためゼロ行列生成
            X_train = np.zeros((num_of_data, N_col))
            # 学習データに対するラベルを格納するためのゼロ行列生成
            y_train = np.zeros((num_of_data))

            # リコネクションがない画像を行列に読み込む
            for idx, path in enumerate(path_list[:10]):
                X_train[idx, :] = cluster.load_regularize(path)
                y_train[idx] = 0  # リコネクションがないことを表すラベル

            # 処理の開始
            X_train_pca = cluster.PCA(X_train)
            cluster_labels = cluster.KMeans(X_train_pca)
            df_re = cluster.save_result(cluster_labels, path_list, dataset)
            # display(df_re)

        logger.debug("END", extra={"addinfo": f"{val_param}\n"})
    logger.debug("END", extra={"addinfo": f"snap{dataset}\n"})


if __name__ == "__main__":
    doClustering()
