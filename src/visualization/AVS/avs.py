# -*- coding utf-8, LF -*-

import os
import sys
from glob import glob
from logging import getLogger

import cv2
import numpy as np

sys.path.append(os.getcwd() + "\src")

from config.params import AVS_IN_DIR, AVS_OUT_DIR
from Visualization.Visualize.SnapData import SnapData


class AvsMethod(SnapData):
    logger = getLogger("res_root").getChild(__name__)

    def __init__(self) -> None:
        super().__init__()

    def _AVSlat2bilat(self, data_name, xy):
        data_name = str(data_name)

        if data_name == "77":
            start = np.array([40, 120])
            end = np.array([4759, 879])
        else:
            print("このデータにはまだ対応していない。AVSで有効な座標を調べよ")
            return None

        xy = np.array(xy)
        bixy = np.array([513, 1025])
        len = end - start + 1
        res = [(xy[0] - start[0]) * (bixy[0] / len[0]), (xy[1] - start[1]) * (bixy[1] / len[1])]
        return res

    def _change_sep_x(self, xrange: list, data_name: int):
        res = [0, 0]
        for i in range(2):
            res[i] = self._AVSlat2bilat(data_name, [xrange[i], 0])[0]
        return res

    def _change_sep_y(self, xrange: list, data_name: int):
        res = [0, 0]
        for i in range(2):
            res[i] = self._AVSlat2bilat(data_name, [0, xrange[i]])[1]
        return res

    def avs(self, in_dir, out_dir, avs_sep, sepy):
        for path in glob(f"{in_dir}*.jpg"):
            # for path in [f"{in_dir}img00_00554.jpg"]:
            print(path)
            im = cv2.imread(path)
            name = os.path.basename(path)
            for s in range(len(avs_sep)):
                separated_im = im[sepy[0] : sepy[1], avs_sep[s][0] : avs_sep[s][1]]
                cv2.imwrite(out_dir + f"{s}/{s}_{name}", separated_im)

        for s in range(len(avs_sep)):
            for path in glob(f"{out_dir}{s}"):
                f = open(f"{path}/description_{s}.txt", mode="w")
                f.write(
                    f"このファイルは{os.path.basename(os.path.dirname(in_dir))}{im.shape}を\n X{avs_sep[s][0]}:{avs_sep[s][1]}\n Y{sepy[0]}:{sepy[1]}\nで切り取った"
                )
                f.close()

        tempsepx = list(map(self._change_sep_x, avs_sep, 77))
        tempsepy = list(map(self._change_sep_y, sepy, 77))

        return tempsepx, tempsepy


# snap77 を分ける、AVSとの紐付け
if __name__ == "__main__":
    # o点が見えるように分けた 77AVSsplit1
    avs_sep = [[120, 440], [380, 700], [630, 950], [890, 1210], [1200, 1520], [1460, 1780], [1720, 2040], [1980, 2300], [2240, 2560]]
    # X点が見えるように分けた 77AVSsplit2
    # avs_sep = [[270, 590], [520, 840], [770, 1090], [1000, 1320], [1320, 1640], [1580, 1900], [1890, 2210], [2100, 2420], [2360, 2680]]
    sepy = [330, 650]

    datasets = [77, 497, 4949]
    xo = ["x", "o"]

    in_dir = AVS_OUT_DIR + "\\77AVS"
    out_dir = AVS_OUT_DIR + "\\AVS_77_x"
