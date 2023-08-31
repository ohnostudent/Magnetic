# -*- coding utf-8, LF -*-

"""

処理に関する共通のパラメータを定義するファイル

"""

import os
import sys

ROOT_DIR = os.getcwd()

BIN_PATH = ROOT_DIR + "/BIN"
IMAGE_PATH = ROOT_DIR + "/images"
LOG_DIR = ROOT_DIR + "/logs"
SNAP_PATH = ROOT_DIR + "/snaps"
SRC_PATH = ROOT_DIR + "/src"

AVS_IN_DIR = IMAGE_PATH + "/AVS"
AVS_OUT_DIR = IMAGE_PATH + "/AVSsplit"

ML_DIR = ROOT_DIR + "/ML"
ML_DATA_DIR = ML_DIR + "/data"
ML_MODEL_DIR = ML_DIR + "/models"
ML_RESULT_DIR = ML_DIR + "/result"


datasets  = [77, 497, 4949]
sides = ["left", "right"]
variable_parameters = ["density", "enstrophy", "magfieldx", "magfieldy", "magfieldz", "pressure", "velocityx", "velocityy", "velocityz"]
labels = ["n", "x", "o"]


def set_dataset(dataset: str):
    if dataset.isnumeric():
        dataset_int = int(dataset)
        if dataset_int not in datasets:
            sys.exit()

    else:
        sys.exit()

    return dataset_int
