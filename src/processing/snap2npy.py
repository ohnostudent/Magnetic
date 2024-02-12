# -*- coding utf-8, LF -*-

import os
import sys

import numpy as np

sys.path.append(os.getcwd() + "/src")

from visualization.visualize.SnapData import SnapData


def snap2npy(sp: SnapData, path: str, dataset: int) -> None:
    """
    snapファイルを左右に分割する

    Args:
        sp (SnapData): _description_
        path (str): _description_
        dataset (int): _description_
    """
    data = sp.loadSnapData(path)
    # print(data.shape)

    _snap_all(path, dataset, data)
    _snap_half_left(path, dataset, data)
    _snap_half_right(path, dataset, data)


def _snap_half_left(path: str, dataset: int, data) -> None:
    # 保存先のpath の作成
    out_path_half = path.replace(f"snap{dataset}", f"half_left/snap{dataset}")
    if not os.path.exists(os.path.dirname(out_path_half)):
        os.makedirs(os.path.dirname(out_path_half))

    # 縦625 * 横256 に切り取る
    half = data[200 : 1025 - 200, :257]

    # numpy 配列として保存
    np.save(out_path_half, half)


def _snap_half_right(path: str, dataset: int, data) -> None:
    # 保存先のpath の作成
    out_path_half = path.replace(f"snap{dataset}", f"half_right/snap{dataset}")
    if not os.path.exists(os.path.dirname(out_path_half)):
        os.makedirs(os.path.dirname(out_path_half))

    # 縦625 * 横256 に切り取る
    half = data[200 : 1025 - 200, 257:]
    # numpy 配列として保存
    np.save(out_path_half, half)


def _snap_all(path: str, dataset: int, data) -> None:
    # 保存先のpath の作成
    out_path_all = path.replace(f"snap{dataset}", f"all/snap{dataset}")
    if not os.path.exists(os.path.dirname(out_path_all)):
        os.makedirs(os.path.dirname(out_path_all))

    # numpy 配列として保存
    np.save(out_path_all, data)


def doSnap2npy(dataset: int) -> None:
    from glob import glob
    from logging import getLogger

    from config.params import DATASETS, SNAP_PATH, VARIABLE_PARAMETERS

    logger = getLogger("Snap_to_npy").getChild("main")

    if dataset not in DATASETS:
        logger.debug("ERROR", extra={"addinfo": f"cannot use dataset{dataset}"})
        return

    sp = SnapData()
    logger.debug("START", extra={"addinfo": f"Snap{dataset} 開始"})

    for param in VARIABLE_PARAMETERS:
        logger.debug("START", extra={"addinfo": f"{param} 開始"})

        for path in glob(SNAP_PATH + f"/snap{dataset}/{param}/*/*"):
            # print(path)
            snap2npy(sp, path, dataset)

        logger.debug("END", extra={"addinfo": f"{param} 終了"})
    logger.debug("END", extra={"addinfo": f"Snap{dataset} 終了"})


def main() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from config.SetLogger import logger_conf

    logger = logger_conf("Snap_to_npy")
    logger.debug("START", extra={"addinfo": "処理開始"})

    with ThreadPoolExecutor(max_workers=6) as exec:
        exec.submit(doSnap2npy, 77)
        exec.submit(doSnap2npy, 497)
        exec.submit(doSnap2npy, 4949)

    logger.debug("END", extra={"addinfo": "処理終了"})


if __name__ == "__main__":
    main()
