# -*- coding utf-8, LF -*-

import json
import os
import sys
from glob import glob
from logging import getLogger

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.getcwd() + "/src")

from config.params import CNN_IMAGE_SHAPE, IMAGE_PATH, IMG_SHAPE, LABELS, ML_DATA_DIR, NPY_SHAPE, SIDES, SNAP_PATH, TRAIN_SHAPE


class _Kernel:
    """
    rad = np.arccos(u/np.sqrt(u**2+v**2))
    color2 = np.array(v) / np.array(u)
    color2 = color2 - min(color2.flat)
    color2 = color2/max(color2.flat)
    speed = np.sqrt(u**2 + v**2)
    lw = 7*speed / speed.max()
    energy = dens * (vX**2 + vY**2) / 2
    """

    def kernel_listxy(self, im1, im2) -> np.ndarray:
        res = np.empty([im1.shape[0], im1.shape[1]])
        res = [[[] for _ in range(im1.shape[1])] for _ in range(im1.shape[0])]
        for x in range(im1.shape[1]):
            for y in range(im1.shape[0]):
                res[y][x] = [im1[y][x], im2[y][x]]
        return np.array(res)

    def kernel_xy(self, im1, im2) -> np.ndarray:
        res = np.zeros([im1.shape[0], im1.shape[1] * 2])
        for x in range(im1.shape[1]):
            for y in range(im1.shape[0]):
                res[y][x * 2] = im1[y][x]
                res[y][x * 2 + 1] = im2[y][x]
        return res

    def kernel_Energy(self, vx, vy, dens):
        return dens * (vx**2 + vy**2) / 2

    def kernel_Rad(self, vx, vy):
        return np.arccos(vx / np.sqrt(vx**2 + vy**2))


class CreateTrain(_Kernel):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.logger = getLogger("fusion").getChild("Create_Train")

    def cut_and_save_from_csv(self, val_param: str, df: pd.Series) -> None:
        """
        ラベリング時の座標をもとに、元データを切り取る

        Args:
            val_param (str): 対象の変数
            df (pd.Series): 切り取った座標データ
        """
        # パラメータ設定
        para, job, side = df["para"], df["job"], df["side"]
        centerx, xlow, xup = df["centerx"], int(df["xlow"]), int(df["xup"])
        centery, ylow, yup = df["centery"], int(df["ylow"]), int(df["yup"])
        label = df["label"]

        if xlow < 0:
            xlow = 0
        if ylow <= 0:
            ylow = 0

        # 画像の読み込み
        npy_path = SNAP_PATH + f"/half_{side}/snap{self.dataset}/{val_param}/{job:02d}/{val_param}.{para:02d}.{job:02d}.npy"  # ファイルパス
        if not os.path.exists(npy_path):
            print(npy_path)

        img = np.load(npy_path)  # 画像データの読み込み
        separated_im = img[ylow:yup, xlow:xup]  # 切り取り

        # 保存
        base_path = ML_DATA_DIR + f"/snap{self.dataset}/point_{LABELS[label]}/{val_param}" + f"/{val_param}_{self.dataset}_{side}.{para:02d}.{job:02d}_{centerx:03d}.{centery:03d}"
        self.save_Data(separated_im, base_path)

    def cut_and_save_from_json(self, path: str, side: str, label: int, val_param: str):
        """
        json の座標データを基に、データを切り取る

        Args:
            path (str): 画像パス
            side (str): left / right
            label (int): 0(n), 1(x), 2(o)
            val_param (str): 対象のパラメータ
        """

        self.logger.debug("START", extra={"addinfo": f"val_param={val_param}, side={side}, label={LABELS[label]}"})

        # 座標データの設定
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        locs = data[str(self.dataset)][side][LABELS[label]]

        # ループ範囲
        if label == 0:
            _job_range = self._set_job_range()

        else:
            _job_range = locs.keys()

        npy_base_path = SNAP_PATH + f"/half_{side}/snap{self.dataset}/{val_param}"

        for param in range(1, 75):
            for job in _job_range:
                npy_path = npy_base_path + f"/{job}/{val_param}.{param :02d}.{job}.npy"

                if not os.path.exists(npy_path):
                    break

                try:
                    npy_save_base_path = ML_DATA_DIR + f"/snap_files/snap{self.dataset}/point_{LABELS[label]}/{val_param}"

                    if not os.path.exists(npy_save_base_path):
                        os.makedirs(npy_save_base_path)

                    npy_img = np.load(npy_path)

                    # 切り取る座標の辞書
                    center = locs[job]["center"]
                    x_range = locs[job]["x_range"]
                    y_range = locs[job]["y_range"]

                    for num in center.keys():
                        centerx, centery = center[num]
                        x_range_low, x_range_up = x_range[num]
                        y_range_low, y_range_up = y_range[num]

                        base_path = npy_save_base_path + f"/{val_param}_{self.dataset}_{side}.{param :02d}.{job}_{centerx :03d}.{centery :03d}"
                        self.cut_binary(npy_img, x_range_low, x_range_up, y_range_low, y_range_up, base_path)

                except ValueError as e:
                    self.logger.error("ERROR", extra={"addinfo": str(e)})

                except cv2.error as e:
                    self.logger.error("ERROR", extra={"addinfo": f"{e}"})

        self.logger.debug("END", extra={"addinfo": f"val_param={val_param}, side={side}, label={LABELS[label]}"})

    def cut_and_save_from_image(self, path: str, side: str, label: int):
        """
        json の座標データを基に、データを切り取る

        Args:
            path (str): 画像パス
            side (str): left / right
            label (int): 0(n), 1(x), 2(o)
        """

        self.logger.debug("START", extra={"addinfo": f"side={side}, label={LABELS[label]}"})
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        locs = data[str(self.dataset)][side][LABELS[label]]
        if label == 0:
            _job_range = self._set_job_range()

        else:
            _job_range = locs.keys()

        lic_base_path = IMAGE_PATH + f"/LIC/snap{dataset}/{side}"

        for param in range(1, 75):
            for job in _job_range:
                lic_path = lic_base_path + f"/lic_snap{dataset}.{side}.magfield.{param :02d}.{job}.bmp"

                if not os.path.exists(lic_path):
                    break

                try:
                    img_save_base_path = ML_DATA_DIR + f"/cnn/point_{LABELS[label]}"

                    if not os.path.exists(img_save_base_path):
                        os.makedirs(img_save_base_path)

                    bmp_img = cv2.imread(lic_path)

                    center = locs[job]["center"]
                    x_range = locs[job]["x_range"]
                    y_range = locs[job]["y_range"]

                    for num in center.keys():
                        centerx, centery = center[num]
                        x_range_low, x_range_up = x_range[num]
                        y_range_low, y_range_up = y_range[num]

                        save_path = img_save_base_path + f"/snap{dataset}_{side}.{param :02d}.{job}_{centerx :03d}.{centery :03d}.bmp"
                        self.cut_images(bmp_img, x_range_low, x_range_up, y_range_low, y_range_up, save_path)

                except ValueError as e:
                    self.logger.error("ERROR", extra={"addinfo": str(e)})

                except cv2.error as e:
                    self.logger.error("ERROR", extra={"addinfo": f"{e}"})

        self.logger.debug("END", extra={"addinfo": f"side={side}, label={LABELS[label]}"})

    def cut_binary(self, img, x_range_low: int, x_range_up: int, y_range_low: int, y_range_up: int, save_path: str) -> None:
        """
        binary ファイルから教師用データを切り取る

        Args:
            img (_type_): _description_
            x_range_low (int): _description_
            x_range_up (int): _description_
            y_range_low (int): _description_
            y_range_up (int): _description_
            save_path (str): _description_
        """
        separated_im = img[y_range_low:y_range_up, x_range_low:x_range_up]  # 切り取り
        img_resize = cv2.resize(separated_im, TRAIN_SHAPE, interpolation=cv2.INTER_LANCZOS4)  # サイズ変更 -> (100, 10)
        self.save_Data(img_resize, save_path)

    def cut_images(self, img, x_range_low: int, x_range_up: int, y_range_low: int, y_range_up: int, save_path: str, reshape_size: int = 15) -> None:
        """
        CNN用に画像を切り取り、正方形にする

        Args:
            img (_type_): 画像データ
            x_range_low (int): _description_
            x_range_up (int): _description_
            y_range_low (int): _description_
            y_range_up (int): _description_
            save_path (str): 保存パス
            reshape_size (int, optional): 画像サイズの補正. Defaults to 15.
        """
        img_range_low_X = int(x_range_low / NPY_SHAPE[0] * IMG_SHAPE[0])
        img_range_low_Y = int(y_range_low / NPY_SHAPE[1] * IMG_SHAPE[1])
        img_range_up_X = int(x_range_up / NPY_SHAPE[0] * IMG_SHAPE[0])
        img_range_up_Y = int(y_range_up / NPY_SHAPE[1] * IMG_SHAPE[1])
        rangeX = img_range_up_X - img_range_low_X
        rangeY = img_range_up_Y - img_range_low_Y
        range_diff = abs(rangeY - rangeX) // 2

        # 正方形にする
        if rangeX > rangeY:
            img_range_low_Y -= range_diff
            img_range_up_Y += range_diff
        else:
            img_range_low_X -= range_diff
            img_range_up_X += range_diff

        # 座標範囲のリサイズ
        if img_range_low_X < 0:
            img_range_up_X += abs(img_range_low_X)
            img_range_low_X = 0

        if img_range_up_X > IMG_SHAPE[0]:
            img_range_low_X -= img_range_up_X - IMG_SHAPE[0]
            img_range_up_X = IMG_SHAPE[0]

        if img_range_low_Y < 0:
            img_range_up_Y += abs(img_range_low_Y)
            img_range_low_Y = 0

        if img_range_up_Y > IMG_SHAPE[1]:
            img_range_low_Y -= img_range_up_Y - IMG_SHAPE[1]
            img_range_up_Y = IMG_SHAPE[1]

        # 教師データの切り取り
        img_cut = img[img_range_low_Y + reshape_size : img_range_up_Y - reshape_size, img_range_low_X + reshape_size : img_range_up_X - reshape_size, :]
        # 正方形のサイズを統一する
        img_cut = cv2.resize(img_cut, CNN_IMAGE_SHAPE)
        # 保存
        cv2.imwrite(save_path, img_cut)

    def _set_job_range(self) -> list:
        """
        使用するデータセットによってjob の範囲が異なる

        Returns:
            list: job list
        """
        if self.dataset == 77:
            job_range = map(lambda x: f"{x :02d}", range(10, 15))
        elif self.dataset == 497:
            job_range = map(lambda x: f"{x :02d}", range(10, 15))
        else:
            job_range = map(lambda x: f"{x :02d}", range(9, 15))
        return list(job_range)

    def loadBinaryData(self, img_path: str, val_params: str) -> list:
        """
        複数の変数データのロード

        Args:
            img_path (str): 画像パス
            val_params (str): 使用変数

        Returns:
            list: 合成データ
        """
        im_list = list(map(lambda val: np.load(img_path.replace(val_params[0], val)), val_params))

        return im_list

    def save_Data(self, result_img, out_path) -> None:
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        np.save(out_path, result_img)

    def create_training(self, kernel, val_params: str, in_path: str, out_path: str):
        im_list = self.loadBinaryData(in_path, val_params)  # 混合データのロード
        result_img = kernel(*im_list)  # データの作成
        self.save_Data(result_img, out_path)  # データの保存


def save_split_data_from_csv(dataset: int) -> None:
    logger = getLogger("fusion").getChild("Split_from_csv")
    csv_path = ML_DATA_DIR + f"/LIC_labels/label_snap{dataset}_org.csv"
    logger.debug("PROCESS", extra={"addinfo": f"Make Train, path={csv_path}"})

    if not os.path.exists(csv_path):
        logger.debug("ERROR", extra={"addinfo": "ファイルが見つかりませんでした"})
        raise ValueError("File not found")

    logger.debug("START", extra={"addinfo": f"{dataset}"})
    md = CreateTrain(dataset)
    df = pd.read_csv(csv_path, encoding="utf-8")
    df_snap: pd.DataFrame = df[df["dataset"] == dataset]

    for val in ["magfieldx", "magfieldy", "velocityx", "velocityy", "density"]:
        logger.debug("START", extra={"addinfo": val})

        for _, d in df_snap.iterrows():
            md.cut_and_save_from_csv(val, d)

        logger.debug("END", extra={"addinfo": val})
    logger.debug("END", extra={"addinfo": f"{dataset}"})


def save_split_data_from_json(dataset: int):
    logger = getLogger("fusion").getChild("Split_from_json")
    path = ML_DATA_DIR + "/LIC_labels/snap_labels.json"
    logger.debug("PROCESS", extra={"addinfo": f"Cut, path={path}"})

    if not os.path.exists(path):
        logger.debug("ERROR", extra={"addinfo": "ファイルが見つかりませんでした"})
        raise ValueError("File not found")

    logger.debug("START", extra={"addinfo": f"{dataset}"})
    md = CreateTrain(dataset)
    for side in SIDES:
        for label in LABELS.keys():
            # for val in ["magfieldx", "magfieldy", "velocityx", "velocityy", "density"]:
            #     md.cut_and_save_from_json(path, side, label, val)
            md.cut_and_save_from_image(path, side, label)

    logger.debug("END", extra={"addinfo": f"{dataset}"})


def makeTrainingData(dataset: int, props_params: list | None = None) -> None:
    logger = getLogger("fusion").getChild("Make_Training")
    logger.debug("START", extra={"addinfo": "Make Training Data"})

    md = CreateTrain(dataset)
    OUT_DIR = ML_DATA_DIR + f"/snap_files/snap{dataset}"
    if props_params is None:
        props_params = [
            (["magfieldx", "magfieldy"], "mag_tupledxy", md.kernel_listxy),
            (["velocityx", "velocityy", "density"], "energy", md.kernel_Energy),
        ]

    # /images/0131_not/density/density_49.50.8_9.528
    for val_params, out_basename, kernel in props_params:
        logger.debug("START", extra={"addinfo": val_params})

        for label in LABELS.keys():  # n, x, o
            logger.debug("START", extra={"addinfo": f"label : {LABELS[label]}"})
            npys_path = OUT_DIR + f"/point_{LABELS[label]}"

            for img_path in glob(npys_path + f"/{val_params[0]}/*.npy"):  # ./ML/data/snap_files/snap4949/point_o/magfieldx/magfieldx_4949_left.01.10_003.351.npy'
                # 保存先のパスの作成
                # ./ML/data/snap_files/snap{dataset}/point_{label}/{out_basename}/{out_basename}_{dataset}_{side}.{param}.{job}_{centerx}.{centery}.npy
                # ./ML/data/snap_files/snap4949/point_o/energy/energy_77_left.01.03_131.543.npy
                out_path = npys_path + f"/{out_basename}/{os.path.basename(img_path).replace(val_params[0], out_basename)}"
                md.create_training(kernel, val_params, img_path, out_path)

            logger.debug("END", extra={"addinfo": f"label : {LABELS[label]}\n"})
        logger.debug("END", extra={"addinfo": f"{val_params}\n\n"})
    logger.debug("END", extra={"addinfo": "Make Training Data\n\n"})


if __name__ == "__main__":
    from config.params import DATASETS
    from config.SetLogger import logger_conf

    logger = logger_conf("fusion")

    for dataset in DATASETS:
        # save_split_data_from_csv(dataset)
        save_split_data_from_json(dataset)
        # makeTrainingData(dataset)
