# -*- coding utf-8, LF -*-

import numpy as np
import os
import sys
sys.path.append(os.getcwd() + "/src")
from Visualization.SnapData import SnapData


def snap2npy(sp: SnapData, path: str, dataset: int):
    data  = sp.loadSnapData(path)
    # print(data.shape)

    _snap_all(path, dataset, data)
    _snap_half(path, dataset, data)


def _snap_half(path, dataset, data):
    out_path_half = path.replace(f"snap{dataset}", f"half/snap{dataset}")
    if not os.path.exists(os.path.dirname(out_path_half)):
        os.makedirs(os.path.dirname(out_path_half))

    half = data[256: 1025-256, :]
    np.save(out_path_half, half)


def _snap_all(path, dataset, data):
    out_path_all = path.replace(f"snap{dataset}", f"all/snap{dataset}")
    if not os.path.exists(os.path.dirname(out_path_all)):
        os.makedirs(os.path.dirname(out_path_all))

    np.save(out_path_all, data)


def doSnap2npy():
    from glob import glob
    from params import SNAP_PATH, datasets, variable_parameters
    from SetLogger import logger_conf


    sp = SnapData()
    logger = logger_conf()
    logger.debug("START", extra={"addinfo": f"処理開始"})

    try:
        for dataset in datasets:
            logger.debug("START", extra={"addinfo": f"Snap{dataset} 開始"})

            for param in variable_parameters:
                logger.debug("START", extra={"addinfo": f"{param} 開始"})

                for path in glob(SNAP_PATH + f"/snap{dataset}/{param}/*/*"):
                    # print(path)
                    snap2npy(sp, path, dataset)

                logger.debug("END", extra={"addinfo": f"{param} 終了"})
            logger.debug("END", extra={"addinfo": f"Snap{dataset} 終了"})
    
    except KeyboardInterrupt:
        logger.debug("END", extra={"addinfo": f"{param} 中断"})
        logger.debug("END", extra={"addinfo": f"Snap{dataset} 中断\n"})
    
    except Exception as e:
        logger.debug(str(e), extra={"addinfo": "\n"})
        logger.debug("END", extra={"addinfo": f"{param} 中断"})
        logger.debug("END", extra={"addinfo": f"Snap{dataset} 中断\n"})
    
    finally:
        logger.debug("END", extra={"addinfo": f"処理終了"})


if __name__ == "__main__":
    doSnap2npy()
