

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from params import ROOT_DIR, IMGOUT


class SnapData():
    def __init__(self) -> None:
        pass

    @classmethod
    def setSnapData(cls, file_path, z=3):
        # データのインポート
        # r : 読み込み, b : バイナリモード
        # if os.path.splitext(file_path)[1] == "npy": # 拡張子の判定
        if file_path[-3:] == "npy": # 拡張子なしの場合を考慮するとこの形になる？
            cls.target, cls.param, cls.job, _ = os.path.basename(file_path).split(".")
            return np.load(file_path)

        cls.target, cls.param, cls.job = os.path.basename(file_path).split(".")        
        with open(file_path, mode="rb") as f:
            if z == 1:
                cls.snap_data = np.fromfile(f, dtype='f', sep='').reshape(1025, 513)

            elif z == 3:
                cls.snap_data = np.fromfile(f, dtype='f', sep='')[:525825].reshape(1025, 513)

            f.close()

        return cls()


    def drawHeatmap(self, viz, saveimg=False, bar_range=None):
        self._makedir()
        # フォルダの作成
        if not os.path.exists(IMGOUT + f"\\{viz}\\{self.target}\\{self.job :02d}"):
            os.makedirs(IMGOUT + f"\\{viz}\\{self.target}\\{self.job :02d}")
            # print(IMGOUT + f"\\{viz}\\{self.target}\\{self.job :02d}")

        # 描画
        if bar_range:
            sns.heatmap(self.snap_data, vmin=bar_range[0], vmax=bar_range[1])
        else:
            sns.heatmap(self.snap_data)
        # 保存
        if saveimg:
            plt.savefig(IMGOUT + f"\\{viz}\\{self.target}\\{self.job :02d}\\{self.target}.{self.param :02d}.{self.job :02d}.png")

        # メモリの開放
        # plt.clf()
        # plt.close() 

    def _makedir(self, target) -> None:
        if not os.path.exists(ROOT_DIR + f"\\{target}"):
            os.makedirs(ROOT_DIR + f"\\{target}")
    