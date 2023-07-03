# -*- coding utf-8, LF -*-

import os
import sys
import numpy as np
sys.path.append(os.getcwd())

from src.params import IMGOUT


class SnapData():
    """
    _convolute, _ave_carnel, _calc -> Visualize.py, /k-means/Clustering.py にて使用

    """
    
    def loadSnapData(self, file_path, z=3):
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
        if not os.path.exists(IMGOUT + f"/{path}"):
            os.makedirs(IMGOUT + f"/{path}")
        

    #畳み込み
    def _convolute(self, data: np.array, carnel: np.array, padding=0, stride=1):
        """
        畳み込み演算。
        """
        if padding:
            print("0パディングの処理未実装  padding=0で実行します")
            padding = 0
        
        c_width = carnel.shape[1]
        c_height = carnel.shape[0]
        result_width = int((data.shape[1] + 2*padding - carnel.shape[1]) / stride + 1)
        result_height = int((data.shape[0] + 2*padding - carnel.shape[0]) / stride + 1)
        convoluted = np.zeros((result_height, result_width))

        orgY = 0
        for resultY in range(result_height):
            orgX = 0
            for resultX in range(result_width):
                array = data[orgY : orgY + c_height, orgX : orgX + c_width]
                # a = convoluted[resultY]
                convoluted[resultY][resultX] = self._calc(array, carnel)
                orgX += stride
            orgY += stride

        return convoluted
    
    
    def _ave_carnel(self, size:int):
        """
        畳み込みにおける平滑化のカーネル作成
        """
        ones = np.ones((size,size))
        res = ones / (size**2)
        return res
    
    def _calc(self, array, carnel):
        result = sum(array * carnel)
        result = sum(result.flat)
        return result
    