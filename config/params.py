# -*- coding utf-8, LF -*-

"""

処理に関する共通のパラメータを定義するファイル

"""

import os
import sys

ROOT_DIR = os.getcwd()
SNAP_PATH = ROOT_DIR + "/snaps"
IMGOUT = ROOT_DIR + "/imgout"
LOG_DIR = ROOT_DIR + "/logs"
SRC_PATH = ROOT_DIR + "/src"
ETC_PATH = ROOT_DIR + "/etc"

AVS_IN_DIR = IMGOUT + "/AVS"
AVS_OUT_DIR = IMGOUT + "/AVSsplit"

ML_DATA_DIR = ROOT_DIR + "/MLdata"
ML_RESULT_DIR = ROOT_DIR + "/MLres"

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
