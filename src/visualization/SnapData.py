# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
sys.path.append(os.getcwd())

from src.params import IMGOUT


class SnapData():
    def setSnapData(self, file_path, z=3):
        # データのインポート
        # r : 読み込み, b : バイナリモード
        # if os.path.splitext(file_path)[1] == "npy": # 拡張子の判定
        if file_path[-3:] == "npy": # 拡張子なしの場合を考慮するとこの形になる？
            self.target, self.param, self.job, _ = map(lambda x: int(x) if x.isnumeric() else x, os.path.basename(file_path).split('.'))
            return np.load(file_path)

        self.target, self.param, self.job = map(lambda x: int(x) if x.isnumeric() else x, os.path.basename(file_path).split('.'))
        with open(file_path, mode="rb") as f:
            if z == 1:
                snap_data = np.fromfile(f, dtype='f', sep='').reshape(1025, 513)
            elif z == 3:
                snap_data = np.fromfile(f, dtype='f', sep='')[:525825].reshape(1025, 513)

            f.close()
        return snap_data

    def makedir(self, path) -> None:
        if not os.path.exists(IMGOUT + f"\{path}"):
            os.makedirs(IMGOUT + f"\{path}")
