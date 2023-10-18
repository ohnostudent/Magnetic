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
LABELS = {0: "n", 1: "x", 2: "o"}
IMAGE_SHAPE = (10, 100)  # (X, Y)
ML_PARAM_DICT = {
    "KMeans": {"n_clusters": 3, "n_init": 10, "max_iter": 300, "tol": 1e-04, "random_state": 100},
    "kneighbors": {"n_clusters": 3, "n_init": 10, "max_iter": 300, "tol": 1e-04, "randomstate": 100},
    "linearSVC": {"C": 0.3, "randomstate": 0},
    "rbfSVC": {"C": 0.3, "gamma": 3, "cls": "ovo", "randomstate": 0},
    "XGBoost": {"colsample_bytree": 0.4, "early_stopping_rounds": 100, "eval_metric": "auc", "learning_rate": 0.02, "max_depth": 4, "missing": -1, "n_estimators": 500, "subsample": 0.8, "params": {}},
}


def set_dataset(dataset: str):
    if dataset.isnumeric():
        dataset_int = int(dataset)
        if dataset_int not in DATASETS:
            sys.exit()

    else:
        sys.exit()

    return dataset_int
