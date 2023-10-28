# -*- coding utf-8, LF -*-

"""

処理に関する共通のパラメータを定義するファイル

"""

import os
import sys

ROOT_DIR = os.getcwd()

BIN_PATH = ROOT_DIR + "/cmd"
IMAGE_PATH = ROOT_DIR + "/images"
LOG_DIR = ROOT_DIR + "/logs"
SNAP_PATH = ROOT_DIR + "/snaps"
SRC_PATH = ROOT_DIR + "/src"

ML_DIR = ROOT_DIR + "/ML"
ML_DATA_DIR = ML_DIR + "/data"
ML_MODEL_DIR = ML_DIR + "/models"
ML_RESULT_DIR = ML_DIR + "/result"

DATASETS = [77, 497, 4949]
SIDES = ["left", "right"]
VARIABLE_PARAMETERS = ["density", "enstrophy", "magfieldx", "magfieldy", "magfieldz", "pressure", "velocityx", "velocityy", "velocityz"]
VARIABLE_PARAMETERS_FOR_TRAINING = ["density", "energy", "magfieldx", "magfieldy", "mag_tupledxy", "velocityx", "velocityy"]
LABELS = {0: "n", 1: "x", 2: "o"}
TRAIN_SHAPE = (10, 100)  # (X, Y)
IMG_SHAPE = [1792, 569]
NPY_SHAPE = [257, 625]
CNN_IMAGE_SHAPE = [100, 100]

ML_PARAM_DICT = {
    "KMeans": {
        "algorithm": "lloyd",  # 使用するアルゴリズム
        "copy_x": True,  # データのコピー
        "init": "k-means++",
        "max_iter": 300,  # 最大反復回数
        "n_clusters": 8,  # クラスタ数
        "n_init": "warn",
        "random_state": None,  # 乱数の seed値
        "tol": 0.0001,  # 停止基準の許容値
        "verbose": 0,  # 詳細な出力を有効
    },
    "KNeighbors": {
        "algorithm": "auto",  # 最近傍値を計算するアルゴリズム(auto, ball_tree, kd_tree, brute)
        "leaf_size": 30,  # BallTree または KDTree に渡される葉のサイズ
        "metric": "minkowski",  # 距離の計算に使用するメトリック
        "metric_params": None,  # メトリック関数の追加のパラメータ
        "n_jobs": None,  # 実行する並列ジョブの数
        "n_neighbors": 5,  # k の数
        "p": 2,  # Minkowski メトリクスの検出力パラメータ
        "weights": "uniform",  # 重み
    },
    "LinearSVC": {
        "C": 1.0,  # 正則化パラメータ、マージン
        "dual": "auto",  # 双対最適化問題または主最適化問題を解決するアルゴリズム
        "fit_intercept": True,  # 切片を適合させるかどうか
        "intercept_scaling": 1,
        "loss": "squared_hinge",  # 損失関数(hinge, squared_hinge)
        "max_iter": 1000,  # 実行される反復の最大数
        "multi_class": "ovr",  # マルチクラス戦略
        "penalty": "l2",  # ペナルティ
        "random_state": None,  # 乱数の seed値
        "tol": 0.0001,  # 停止基準の許容値
        "verbose": 50,  # 詳細な出力を有効
    },
    "rbfSVC": {
        "C": 1.0,  # 正則化パラメータ、マージン
        "cache_size": 200,  # キャッシュサイズ
        "coef0": 0.0,  # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        "decision_function_shape": "ovr",
        "degree": 3,  # 多項式(poly)カーネルの次数
        "gamma": "scale",  # カーネルの係数、ガウスカーネル(rbf): 1/(n_features * X.var()) と シグモイドカーネル(sigmoid): 1 /n_features
        "kernel": "rbf",  # カーネル('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
        "max_iter": -1,  # ソルバー内の反復に対するハード制限
        "probability": False,  # True の場合、予測時に各クラスに属する確率を返す
        "random_state": None,  # 乱数の seed値
        "shrinking": True,  # 縮小ヒューリスティックを使用するかどうか
        "tol": 0.001,  # 停止基準の許容値
        "verbose": False,  # 詳細な出力を有効
    },
    "XGBoost": {
        "colsample_bytree": None,
        "early_stopping_rounds": None,
        "eval_metric": None,
        "gamma": None,
        "learning_rate": "shrinkage",
        "max_depth": 4,
        "missing": None,
        "n_estimators": 1000,
        "n_jobs": None,
        "subsample": None,
        "tree_method": None,  # 木構造の種類 (hist/gpu_hist)
        "random_state": None,  # 乱数のseed値
        "reg_alpha": None,
        "reg_lambda": None,
        "verbosity": 100,
    },
}


def set_dataset(dataset: str):
    if dataset.isnumeric():
        dataset_int = int(dataset)
        if dataset_int not in DATASETS:
            sys.exit()
    else:
        sys.exit()

    return dataset_int


def dict_to_str(param_dict: dict) -> str:
    param_list_sorted = sorted(param_dict.items())
    return ".".join(map(lambda x: f"{x[0]}={x[1]}", param_list_sorted))
