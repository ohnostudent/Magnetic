# -*- coding utf-8, LF -*-

import os
import sys
from glob import glob
from logging import getLogger

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/src")

from config.params import ML_DATA_DIR, SNAP_PATH, labels
from Processing.kernel import _Kernel


class CreateTrain(_Kernel):
    res = {0: "n", 1: "x", 2: "o"}

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def cut_and_save(self, val_param, df) -> None:
        """
        ラベリング時の座標をもとに、元データを切り取る
        """
        para, job, side = df["para"], df["job"], df["side"]
        centerx, xlow, xup = df["centerx"], int(df["xlow"]), int(df["xup"])
        centery, ylow, yup = df["centery"], int(df["ylow"]), int(df["yup"])
        label = df["label"]

        if xlow < 0:
            xlow = 0
        if ylow <= 0:
            ylow = 0

        npy_path = SNAP_PATH + f"/half_{side}/snap{self.dataset}/{val_param}/{job:02d}/{val_param}.{para:02d}.{job:02d}.npy"
        if not os.path.exists(npy_path):
            print(npy_path)

        img = np.load(npy_path)
        separated_im = img[ylow:yup, xlow:xup]
        print(separated_im.shape)

        base_path = (
            ML_DATA_DIR
            + f"/snap{self.dataset}/point_{self.res[label]}/{val_param}"
            + f"/{val_param}_{self.dataset}_{side}.{para:02d}.{job:02d}_{centerx:03d}.{centery:03d}"
        )
        self.save_Data(separated_im, base_path)

    # 複数の変数を混合したデータを作成する
    def loadBinaryData(self, img_path, val_params) -> list:
        im_list = []
        for val in val_params:
            im = np.load(img_path.replace(val_params[0], val))
            im_list.append(im)

        return im_list

    def save_Data(self, resim, out_path) -> None:
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        np.save(out_path, resim)


def save_split_data(dataset: int) -> None:
    logger = getLogger("res_root").getChild(__name__)

    csv_path = ML_DATA_DIR + f"/LIC_labels/label_snap{dataset}_org.csv"
    if not os.path.exists(csv_path):
        raise ValueError

    md = CreateTrain(dataset)
    df = pd.read_csv(csv_path, encoding="utf-8")
    df_snap: pd.DataFrame = df[df["dataset"] == dataset]

    for val in ["magfieldx", "magfieldy", "velocityx", "velocityy", "density"]:
        logger.debug("START", extra={"addinfo": val})

        for _, d in df_snap.iterrows():
            md.cut_and_save(val, d)

        logger.debug("END", extra={"addinfo": val})


def makeTrainingData(dataset: int) -> None:
    logger = getLogger("res_root").getChild(__name__)

    md = CreateTrain(dataset)
    props_params = [
        (["magfieldx", "magfieldy"], "mag_tupledxy", md.kernellistxy),
        (["velocityx", "velocityy", "density"], "energy", md.kernelEnergy),
    ]
    OUT_DIR = ML_DATA_DIR + f"/snap{dataset}"

    # /images/0131_not/density/density_49.50.8_9.528
    for val_params, out_basename, kernel in props_params:
        logger.debug("START", extra={"addinfo": val_params})
        for label in labels:  # n, o, x
            logger.debug("START", extra={"addinfo": f"label : {label}"})
            npys = OUT_DIR + f"/point_{label}"

            for img_path in glob(npys + "/" + val_params[0] + "/*.npy"):
                im_list = md.loadBinaryData(img_path, val_params)  # 混合データのロード
                resim = kernel(*im_list)  # データの作成

                # 保存先のパスの作成
                # /MLdata/snap{dataset}/{out_basename}/{out_basename}_{dataset}.{param}.{job}_{centerx}.{centery}.npy
                # /MLdata/snap77/energy/energy_77.01.03_131.543.npy
                out_path = npys + "/" + out_basename + "/" + os.path.basename(img_path).replace(val_params[0], out_basename)
                md.save_Data(resim, out_path)  # データの保存

            logger.debug("END", extra={"addinfo": f"label : {label}"})
        logger.debug("END", extra={"addinfo": val_params})


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print()
        sys.exit()
    else:
        dataset = sys.argv[1]

    from config.params import set_dataset
    from config.SetLogger import logger_conf

    logger = logger_conf()
    dataset = set_dataset(dataset)

    logger.debug("START", extra={"addinfo": "Cut"})
    save_split_data(dataset)
    logger.debug("END", extra={"addinfo": "Cut"})

    logger.debug("START", extra={"addinfo": "Make Train"})
    makeTrainingData(dataset)
    logger.debug("END", extra={"addinfo": "Make Train"})
