# -*- coding utf-8, LF -*-

import os

ROOT_DIR = os.getcwd()
SNAP_PATH = ROOT_DIR + "/snaps"
IMGOUT = ROOT_DIR + "/imgout"
LOG_DIR = ROOT_DIR + "/log"
SRC_PATH = ROOT_DIR + "/src"

AVS_IN_DIR = IMGOUT + "/AVS"
AVS_OUT_DIR = IMGOUT + "/AVSsplit"

ML_DATA_DIR = ROOT_DIR + "/MLdata"
ML_RESULT_DIR = ROOT_DIR + "/MLres"

datasets  = [77, 497, 4949]
parameter = ["density", "enstrophy", "magfieldx", "magfieldy", "magfieldz", "pressure", "velocityx", "velocityy", "velocityz"]
