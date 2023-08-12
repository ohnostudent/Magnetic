# -*- coding utf-8, LF -*-

import os
import sys
from glob import glob
from logging import getLogger

import numpy as np
import pandas as pd

from config.params import ML_DATA_DIR, SNAP_PATH, labels
from Processing.kernel import _kernel

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/src")


class crateTrain(_kernel):
    res = {0: "n", 1: "x", 2: "o"}

    def __init__(self) -> None:
        pass

    def cut_and_save(self, dataset, val_param) -> None:
        """
        ラベリング時の座標をもとに、元データを切り取る
        """
        df = pd.read_csv(ML_DATA_DIR + f"/LIC_labels/label_snap{dataset}_all.csv", encoding="utf-8")
        df_snap: pd.DataFrame = df[df["dataset"] == dataset]

        for _, d in df_snap.iterrows():
            para, job, side = d["para"], d["job"], d["side"]
            centerx, xlow, xup = d["centerx"], int(d["xlow"]), int(d["xup"])
            centery, ylow, yup = d["centery"], int(d["ylow"]), int(d["yup"])
            label = d["label"]

            img = np.load(SNAP_PATH + f"/half_{side}/snap{dataset}/{val_param}/{job:02d}/{val_param}.{para:02d}.{job:02d}.npy")
            separated_im = img[ylow:yup, xlow:xup]

            base_path = ML_DATA_DIR + f"/snap{dataset}/point_{self.res[label]}/{val_param}"
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            np.save(base_path + f"/{val_param}_{dataset}.{para:02d}.{job:02d}_{centerx}.{centery}", separated_im)

    # 複数の変数を混合したデータを作成する
    def loadBinaryData(self, impath, val_params):
        im_list = []
        for val in val_params:
            im = np.load(impath.replace(val_params[0], val))
            im_list.append(im)

        return im_list

    def saveFusionData(self, resim, outpath):
        if not os.path.exists(os.path.dirname(outpath)):
            os.mkdir(os.path.dirname(outpath))

        np.save(outpath, resim)


def makeTrainingData(dataset: int) -> None:
    logger = getLogger("res_root").getChild(__name__)

    props_params = [
        (["magfieldx", "magfieldy"], "mag_tupledxy", crateTrain().kernellistxy),
        (["velocityx", "velocityy", "density"], "energy", crateTrain().kernelEnergy),
    ]
    OUT_DIR = ML_DATA_DIR + f"/snap{dataset}"

    md = crateTrain()
    # /imgout/0131_not/density/density_49.50.8_9.528
    for val_params, outbasename, kernel in props_params:
        for a in labels:
            npys = OUT_DIR + f"/point_{a}"

            for impath in glob(npys + "/" + val_params[0] + "/*.npy"):
                im_list = md.loadBinaryData(impath, val_params)  # 混合データのロード
                resim = kernel(*im_list)  # データの作成

                # 保存先のパスの作成
                # /MLdata/snap{dataset}/{outbasename}/{outbasename}_{dataset}.{param}.{job}_{centerx}.{centery}.npy
                # /MLdata/snap77/energy/energy_77.01.03_131.543.npy
                outpath = npys + "/" + outbasename + "/" + os.path.basename(impath).replace(val_params[0], outbasename)
                md.saveFusionData(resim, outpath)  # データの保存


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print()
        sys.exit()
    else:
        dataset = sys.argv[1]

    from config.params import set_dataset

    dataset = set_dataset(dataset)

    # md = crateTrain()
    # for val in ["magfieldx", "magfieldy", "velocityx", "velocityy", "density"]:
    #     md.cut_and_save(dataset, val)

    makeTrainingData(dataset)
