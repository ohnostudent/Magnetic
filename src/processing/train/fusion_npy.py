# -*- coding utf-8, LF -*-

import json
import os
import sys
from glob import glob
from logging import getLogger

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/src")

from config.params import ML_DATA_DIR, SNAP_PATH, labels, sides
from Processing.train.kernel import _Kernel


class CreateTrain(_Kernel):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.logger = getLogger("main").getChild("Create_Train")

    def cut_and_save_from_csv(self, val_param, df) -> None:
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

        base_path = (
            ML_DATA_DIR
            + f"/snap{self.dataset}/point_{labels[label]}/{val_param}"
            + f"/{val_param}_{self.dataset}_{side}.{para:02d}.{job:02d}_{centerx:03d}.{centery:03d}"
        )
        self.save_Data(separated_im, base_path)

    def cut_and_save_from_json(self, path: str, side: str, label: int, val_param: str):
        logger.debug("START", extra={"addinfo": labels[label]})
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        locs = data[str(self.dataset)][side][labels[label]]
        if locs == {}:
            return

        if label == "n":
            job_range = map(str, range(7, 15))
        else:
            job_range = locs.keys()

        npy_base_path = SNAP_PATH + f"/half_{side}/snap{self.dataset}/{val_param}"
        for param in range(1, 31):
            for job in job_range:
                npy_path = npy_base_path + f"/{int(job):02d}/{val_param}.{param:02d}.{int(job):02d}.npy"
                if not os.path.exists(npy_path):
                    break

                try:
                    center = locs[job]["center"]
                    x_range = locs[job]["x_range"]
                    y_range = locs[job]["y_range"]

                    for num in center.keys():
                        centerx, centery = center[num]
                        xlow, xup = x_range[num]
                        ylow, yup = y_range[num]

                        img = np.load(npy_path)
                        separated_im = img[ylow:yup, xlow:xup]

                        base_path = (
                            ML_DATA_DIR
                            + f"/snap{self.dataset}/point_{labels[label]}/{val_param}"
                            + f"/{val_param}_{self.dataset}_{side}.{param:02d}.{int(job):02d}_{centerx:03d}.{centery:03d}"
                        )
                        self.save_Data(separated_im, base_path)

                except ValueError as e:
                    logger.error("ERROR", extra={"addinfo": str(e)})

        logger.debug("END", extra={"addinfo": labels[label]})

    # 複数の変数を混合したデータを作成する
    def loadBinaryData(self, img_path, val_params) -> list:
        im_list = []
        for val in val_params:
            im = np.load(img_path.replace(val_params[0], val))
            im_list.append(im)

        return im_list

    def save_Data(self, result_img, out_path) -> None:
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        np.save(out_path, result_img)


def save_split_data_from_json(dataset: int):
    logger = getLogger("main").getChild("Split_from_json")
    path = ML_DATA_DIR + "/LIC_labels/test.json"
    if not os.path.exists(path):
        raise ValueError

    md = CreateTrain(dataset)
    for val in ["magfieldx", "magfieldy", "velocityx", "velocityy", "density"]:
        logger.debug("START", extra={"addinfo": val})
        for side in sides:
            logger.debug("START", extra={"addinfo": side})
            for label in range(3):
                md.cut_and_save_from_json(path, side, label, val)

            logger.debug("END", extra={"addinfo": side})
        logger.debug("END", extra={"addinfo": val})


def save_split_data_from_csv(dataset: int) -> None:
    logger = getLogger("main").getChild("Split_from_csv")

    csv_path = ML_DATA_DIR + f"/LIC_labels/label_snap{dataset}_org.csv"
    if not os.path.exists(csv_path):
        raise ValueError

    md = CreateTrain(dataset)
    df = pd.read_csv(csv_path, encoding="utf-8")
    df_snap: pd.DataFrame = df[df["dataset"] == dataset]

    for val in ["magfieldx", "magfieldy", "velocityx", "velocityy", "density"]:
        logger.debug("START", extra={"addinfo": val})

        for _, d in df_snap.iterrows():
            md.cut_and_save_from_csv(val, d)

        logger.debug("END", extra={"addinfo": val})


def makeTrainingData(dataset: int) -> None:
    logger = getLogger("main").getChild("Make_Training")

    md = CreateTrain(dataset)
    props_params = [
        (["magfieldx", "magfieldy"], "mag_tupledxy", md.kernel_listxy),
        (["velocityx", "velocityy", "density"], "energy", md.kernel_Energy),
    ]
    OUT_DIR = ML_DATA_DIR + f"/snap{dataset}"

    # /images/0131_not/density/density_49.50.8_9.528
    for val_params, out_basename, kernel in props_params:
        logger.debug("START", extra={"addinfo": val_params})
        for label in labels:  # n, o, x
            logger.debug("START", extra={"addinfo": f"label : {label}"})
            npys_path = OUT_DIR + f"/point_{label}"

            for img_path in glob(npys_path + f"/{val_params[0]}/*.npy"):
                im_list = md.loadBinaryData(img_path, val_params)  # 混合データのロード
                result_img = kernel(*im_list)  # データの作成

                # 保存先のパスの作成
                # /MLdata/snap{dataset}/{out_basename}/{out_basename}_{dataset}.{param}.{job}_{centerx}.{centery}.npy
                # /MLdata/snap77/energy/energy_77.01.03_131.543.npy
                out_path = npys_path + f"/{out_basename}/{os.path.basename(img_path).replace(val_params[0], out_basename)}"
                md.save_Data(result_img, out_path)  # データの保存

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
    # save_split_data_from_csv(dataset)
    save_split_data_from_json(dataset)
    logger.debug("END", extra={"addinfo": "Cut"})

    logger.debug("START", extra={"addinfo": "Make Train"})
    makeTrainingData(dataset)
    logger.debug("END", extra={"addinfo": "Make Train"})
