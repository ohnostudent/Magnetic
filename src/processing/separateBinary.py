# -*- coding: utf-8 -*-

import numpy as np
from glob import glob
import sys
import os
from logging import getLogger
sys.path.append(os.getcwd() + "/src")
from src.Visualization.SnapData import SnapData


def separate_binary(indir, outdir, sep, sepy, targets):
    # ログの取得
    logger = getLogger("res_root").getChild(__name__)
    logger.debug("Main Start", extra={"addinfo": ""})

    sp = SnapData()
    for target in targets:
        logger.debug("TARGET", extra={"addinfo": target})

        for path in glob(indir + f"{target}/*/*"):
            name = os.path.basename(path)
            logger.debug("FILE", extra={"addinfo": name})
            img = sp.setSnapData(path)

            for idx, d in enumerate(sep):
                # ファイル名の変更
                file_name = name.replace(target, f"{target}.{idx}")
                # 分割
                separated_img = img[sepy[0]: sepy[1], d[0]: d[1]]
                # .npy として保存
                np.save(outdir + f"/{target}/{idx}/{file_name}", separated_img)

        # 説明用 .txt の作成
        for s in range(len(sep)):
            for path in glob(outdir + f"{target}/{s}"):
                with open(f"{path}/description_{s}.txt", mode = "w") as f:
                    f.write(f"このファイルは{os.path.basename(os.path.dirname(indir))}{img.shape}を/n X{sep[s][0]}:{sep[s][1]}/n Y{sepy[0]}:{sepy[1]}/nで切り取った")
                    f.close()
        
        logger.debug("FILE", extra={"addinfo": name})
    logger.debug("TARGET END", extra={"addinfo": target})



if __name__ == "__main__":
    from config.params import SNAP_PATH, IMGOUT
    from config.SetLogger import logger_conf
    logger = logger_conf()

    ###############################
    # 
    indir = SNAP_PATH + "/snap77"
    outdir  = IMGOUT + "/snap77split1"
    sep = [[8, 43], [36, 71], [64, 99], [92, 127], [126, 161], [154, 189], [182, 217], [210, 245], [239, 274]]
    ###############################
    # indir = SNAP_PATH + "/snap49"
    # outdir  = IMGOUT + "/snap49split1"
    # sep = [[8, 44], [36, 72], [54, 90], [92, 128], [110, 146], [142, 178], [168, 204], [200, 236], [235, 271]]
    ###############################
    #X点が見えるように分けた 77AVSsplit2
    # indir = SNAP_PATH + "/snap49"
    # outdir  = IMGOUT + "/snap49split2"
    # sep = [[22, 58], [52, 88], [74, 110], [102, 138], [124, 160], [154, 190], [182, 218], [213, 249], [249, 285]]
    ###############################

    targets = ["magfieldx","magfieldy"]
    sepy = [283, 715]

    separate_binary(indir, outdir, sep, sepy, targets)
