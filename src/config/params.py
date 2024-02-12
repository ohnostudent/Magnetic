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
VARIABLE_PARAMETERS_FOR_TRAINING = ["density", "energy", "enstrophy", "pressure", "magfieldx", "magfieldy", "velocityx", "velocityy"]
LABELS = ["n", "x", "o"]
TRAIN_SHAPE = (10, 100)  # (X, Y)
IMG_SHAPE = [1792, 569]
NPY_SHAPE = [257, 625]
CNN_IMAGE_SHAPE = [96, 96]


def set_dataset(dataset: str):
    if dataset.isnumeric():
        dataset_int = int(dataset)
        if dataset_int not in DATASETS:
            sys.exit()
    else:
        sys.exit()

    return dataset_int


def dict_to_str(param_dict: dict, sep=".") -> str:
    param_list_sorted = sorted(param_dict.items())
    return sep.join(map(lambda x: f"{x[0]}={x[1]}", param_list_sorted))
