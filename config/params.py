# -*- coding utf-8, LF -*-

"""

処理に関する共通のパラメータを定義するファイル

"""

import os
import sys

ROOT_DIR = os.getcwd()

BIN_PATH = ROOT_DIR + "/BIN"
IMAGES = ROOT_DIR + "/images"
LOG_DIR = ROOT_DIR + "/logs"
SNAP_PATH = ROOT_DIR + "/snaps"
SRC_PATH = ROOT_DIR + "/src"

AVS_IN_DIR = IMAGES + "/AVS"
AVS_OUT_DIR = IMAGES + "/AVSsplit"

ML_DIR = ROOT_DIR + "/ML"
ML_DATA_DIR = ML_DIR + "/data"
ML_MODEL_DIR = ML_DIR + "/models"
ML_RESULT_DIR = ML_DIR + "/result"


datasets  = [77, 497, 4949]
variable_parameters = ["density", "enstrophy", "magfieldx", "magfieldy", "magfieldz", "pressure", "velocityx", "velocityy", "velocityz"]
labels = ["n", "o", "x"]


def set_dataset(dataset: str):
    from config.params import datasets

    if dataset.isnumeric():
        dataset = int(dataset)
        if dataset not in datasets:
            sys.exit()

    else:
        sys.exit()

    return dataset
