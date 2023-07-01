# -*- coding utf-8, LF -*-

import numpy as np
import os
import sys
sys.path.append(os.getcwd() + "\src")
from Visualization.SnapData import SnapData


def snap2npy(sp: SnapData, path: str, num: int):
    data  = sp.loadSnapData(path)
    # print(data.shape)

    _snap_all(path, num, data)
    _snap_half(path, num, data)


def _snap_half(path, num, data):
    out_path_half = path.replace(f"snap{num}", f"half\snap{num}")
    if not os.path.exists(os.path.dirname(out_path_half)):
        os.makedirs(os.path.dirname(out_path_half))

    half = data[:, :257]
    np.save(out_path_half, half)


def _snap_all(path, num, data):
    out_path_all = path.replace(f"snap{num}", f"all\snap{num}")
    if not os.path.exists(os.path.dirname(out_path_all)):
        os.makedirs(os.path.dirname(out_path_all))

    np.save(out_path_all, data)



if __name__ == "__main__":
    from glob import glob
    from params import SNAP_PATH
    from SetLogger import logger_conf


    sp = SnapData()
    logger = logger_conf()

    for i in [77, 497, 4949]:
        logger.debug("START", extra={"addinfo": f"Snap{i}"})

        for target in ["density", "enstrophy", "magfieldx", "magfieldy", "magfieldz", "pressure", "velocityx", "velocityy", "velocityz"]:
            logger.debug("START", extra={"addinfo": f"{target}"})

            for path in glob(SNAP_PATH + f"\snap{i}\{target}\*\*"):
                # print(path)
                snap2npy(sp, path, i)

            logger.debug("END", extra={"addinfo": f"{target}"})
        logger.debug("END", extra={"addinfo": f"Snap{i}"})

