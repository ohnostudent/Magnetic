# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from SnapData import SnapData
from src.params import IMGOUT, SNAP_PATH


class Visualize(SnapData):
    def __init__(self) -> None:
        super().__init__()
    
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

        orgX = 0
        orgY = 0
        for resultY in range(result_height):
            for resultX in range(result_width):
                array = data[orgY : orgY + c_height, orgX : orgX + c_width]
                # a = convoluted[resultY]
                convoluted[resultY][resultX] = self._calc(array)
                orgX += stride
            orgX = 0
            orgY += stride
        return convoluted
    
    def _calc(self, array, carnel):
        result = sum(array * carnel)
        result = sum(result.flat)
        return result

    def _ave_carnel(self, size:int):
        """
        畳み込みにおける平滑化のカーネル作成
        """
        ones = np.ones((size,size))
        res = ones / (size**2)
        return res

    #離散データの微分
    def _diff(self, x, h):
        """
        精度低めの普通の微分。誤差:h**2
        """
        res = x[2:] - x[:-2]
        # print(x[1:])
        # print(x[:-1])
        return res/(2*h)
    
    def _diff4(self, x, h):
        """
        精度高めの微分・。誤差:h**4
        1回微分{-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)}/12h
        xは時系列データ,hはデータ間の時間(second)
        ベクトル長が4短くなる
        """
        res = -x[4:] + 8*x[3:-1] - 8*x[1:-3] + x[:-4]
        return res/(12*h)
    
    def _diff4_xy(self, data: np.ndarray, h:float, vectol: str):
        """
        diff4を使った行列の横方向偏微分
        """
        if vectol == "y":
            data = data.T

        res = np.ndarray([])
        for vec in data:
            if res.shape == tuple():
                res = self._diff4(vec, h)
            else:
                res = np.append(res, self._diff4(vec, h))
        
        if vectol == "y":
            return res.reshape(data.shape[0],data.shape[1]-4).T
        return res.reshape(data.shape[0],data.shape[1]-4)

    def _rot2d(self, vX: np.ndarray, vY: np.ndarray):
        """
        vX,vYよりローテーションを出す。出力はz方向のベクトル
        y方向については、vZ(zx平面)、vX(zx平面)の順で入力、x方向はvY(yz平面)、vZ(yz平面)の順で入力すると得られる。
        x､y､zそれぞれの出力であるスカラーの行列に対して、
        それぞれの方向の単位ベクトルを掛けて足せば3次元のローテーションが求まる

        rot2dの結果に対して単位ベクトルを掛けるやり方。
        a = np.array([[0,1,2,3],[0,4,5,6],[0,7,8,9]])
        e = np.array([1,0,0])
        mylist = [[j *e for j in i ] for i in a]
        """
        return self._diff4_xy(vY,1)[2:-2,] - self._diff4_xy(vX,1)[:,2:-2]

    # 
    def _calc_radian(self, u,v):
        return np.arccos(u / np.sqrt(u ** 2 + v ** 2))

    # 
    def drawHeatmap(self, path, saveimg=True, bar_range=None):
        # 描画
        snap_data = self.setSnapData(path)
        plt.figure(figsize=(10, 15))
        if bar_range:
            sns.heatmap(snap_data, vmin=bar_range[0], vmax=bar_range[1])
        else:
            sns.heatmap(snap_data)
        
        self._savefig(f"\heatmap\{self.target}", saveimg)

    # エッジの表示
    def drawEdge(self, path, save=True):
        #cv2で扱える0-255の整数に整形
        snap_data = self.setSnapData(path)
        snap_data = (snap_data - min(snap_data.flat))*254 / max(snap_data.flat)
        snap_data = snap_data.astype("uint8")

        plt.figure(figsize=(10, 15))
        edges = cv2.Canny(snap_data, threshold1=150, threshold2=200)
        plt.imshow(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

        if save:
            self.makedir(f"\\visualization\\edges\{self.target}\{self.job :02d}")
            plt.tight_layout()
            cv2.imwrite(IMGOUT + f"\\visualization\\edges\{self.target}\{self.job :02d}\{self.target}.{self.param :02d}.{self.job :02d}.png", edges)
        # メモリの開放
        plt.clf()
        plt.close() 

    # 流線の可視化
    def drawStream(self, X, Y, compress=0):
        # dataX = X[350:700]
        # dataY = Y[350:700]
        dataX = X[350:700, 270:590]
        dataY = Y[350:700, 270:590]
        # dataY = Y[350:700, 224:304]
        # sep = [[270,590],[520,840],[770,1090],[1000,1320],[1320,1640],[1580,1900],[1890,2210],[2100,2420],[2360,2680]]
        # sepy = [330,650]

        #計算が重いので平滑化フィルターの畳み込みで圧縮
        if compress:
            carnel1 = self._ave_carnel(compress)
            carnel2 = carnel1.T
            dataX = self._convolute(dataX, carnel2, stride=compress)
            dataY = self._convolute(dataY, carnel1, stride=compress)

        x = range(dataX.shape[1])
        y = range(dataY.shape[0])
        #X,Y方向それぞれのベクトルに対して座標の行列を設定
        X, Y = np.meshgrid(x, y)
        #X,Y方向それぞれのベクトルの強さ
        u = dataX
        v = dataY
        #########rotの計算途中の微分でデータの端っこが削れる
        rot = self._rot2d(u, v)
        u = u[2:-2,2:-2]
        v = v[2:-2,2:-2]
        X = X[2:-2,2:-2]
        Y = Y[2:-2,2:-2]
        ##########
        color = u ** 2 + v ** 2
        color = color * 2 / max(color.flat)
        rad = np.arccos(u / np.sqrt(u ** 2 + v ** 2))
        color2 = np.array(v) / np.array(u)
        color2 = color2 - min(color2.flat)
        color2 = color2 / max(color2.flat)
        speed = np.sqrt(u ** 2 + v ** 2)
        lw = 7 * speed / speed.max()
        
        fig = plt.figure(1)
        # plt.contour(X,Y,rad)
        # show(rot,bar_range=[-0.05,0.05])
        # sns.heatmap(dataY)
        # strm = plt.streamplot(X, Y, u, v, density=[5], color=color, arrowstyle='-', linewidth=1,cmap="rainbow")
        strm = plt.streamplot(X, Y, u, v, density=[3], color=rot, arrowstyle='-', linewidth=lw,cmap="rainbow")
        # strm = plt.streamplot(X, Y, u, v, density=[0.5], color=rad, arrowstyle='-', linewidth=1,cmap="rainbow", minlength=0.001)
        rad2 = abs(rad - (3.1415927/2))
        # sns.heatmap(rad2, cmap="bone")
        # strm = plt.streamplot(X, Y, u, v, density=[1,5], color=black, arrowstyle='-|>', linewidth=1)
        
        #fig.colorbar(strm.lines)
        # plt._savefig(IMGOUT}1111/{number}.png")
        # plt.show()
    
    # エネルギーの速さと密度について
    def drawEnergy_for_velocity(self, dens_path, vx_path, vy_path, save=True):
        dens_data = self.setSnapData(dens_path)
        vx_data = self.setSnapData(vx_path)
        vy_data = self.setSnapData(vy_path)

        plt.figure(figsize=(40, 20))
        energy = dens_data * (vx_data ** 2 + vy_data ** 2) / 2
        sns.heatmap(energy)

        path = f"\Energy_velocity"
        self._savefig(path, save)

    # エネルギーの磁場について
    def drawEnergy_for_magfield(self, magx_path, magy_path, save=True):
        magx_data = self.setSnapData(magx_path)
        magy_data = self.setSnapData(magy_path)

        plt.figure(figsize=(40, 20))
        
        bhoge = (magx_data ** 2 + magy_data ** 2) / 2
        sns.heatmap(bhoge)

        path = f"\Energy_magfield"
        self._savefig(path, save)

    # das
    def drawStreamHeatmap(self, magx1, magy1, magx2, magy2, save=True):
        dataX1 = magx1[350:700, 270:590]
        dataX2 = magx2[350:700, 270:590]
        dataY1 = magy1[350:700, 270:590]
        dataY2 = magy2[350:700, 270:590]
        dataX = dataX2 - dataX1
        dataY = dataY2 - dataY1

        x = range(dataX.shape[1])
        y = range(dataY.shape[0])

        # X,Y方向それぞれのベクトルに対して座標の行列を設定
        X, Y = np.meshgrid(x, y)

        # X,Y方向それぞれのベクトルの強さ
        u = dataX
        v = dataY

        # rotの計算途中の微分でデータの端っこが削れる
        # rot1 = _rot2d(dataX1, dataY1)
        # rot2 = _rot2d(dataX2, dataY2)
        # rot = rot2 - rot1
        # u = u[2:-2,2:-2]
        # v = v[2:-2,2:-2]
        # X = X[2:-2,2:-2]
        # Y = Y[2:-2,2:-2]
        
        rad1 = self._calc_radian(dataX1, dataY1)
        rad2 = self._calc_radian(dataX2, dataY2)
        rad = rad2 - rad1
        
        ax1 = plt.subplot(2, 1, 1)
        ax1.pcolor(rad, cmap="brg", vmax=0.1, vmin=-0.1)

        ax2 = plt.subplot(2, 1, 2)
        ax2.pcolor(dataX1, vmax = 0.03)

        path = "\StreamHeatmap\snap{i}"
        self._savefig(path, save)

    # 保存
    def _savefig(self, path, save=True):
        # フォルダの作成
        if save:
            self.makedir(f"\\visualization\{path}\{self.job :02d}")
            plt.tight_layout()
            plt.savefig(IMGOUT + f"\\visualization\{path}\{self.job :02d}\{self.target}.{self.param :02d}.{self.job :02d}.png")

        # メモリの開放
        plt.clf()
        plt.close() 



def main():
    from glob import glob

    for i in [77, 497, 4949]:
        target_path = SNAP_PATH + f"\\snap{i}"

        files = {}
        files["density"] = glob(target_path + f"\\density\\*\\*")
        files["velocityx"] = glob(target_path + f"\\velocityx\\*\\*")
        files["velocityy"] = glob(target_path + f"\\velocityy\\*\\*")


        viz = Visualize()
        for dens_path, vx_path, vy_path in zip(files["density"], files["velocityx"], files["velocityy"]):
            viz.drawEnergy_for_velocity(dens_path, vx_path, vy_path)

        files["magfieldx"] = glob(target_path + f"\\magfieldx\\*\\*")
        files["magfieldy"] = glob(target_path + f"\\magfieldy\\*\\*")
        for magx_path, magy_path in zip(files["magfieldx"], files["magfieldy"]):
            viz.drawEnergy_for_magfield(magx_path, magy_path)
            
        files["enstrophy"] = glob(target_path + f"\\enstrophy\\*\\*")
        for target in ["velocityx", "velocityy", "magfieldx", "magfieldy", "density", "enstrophy"]:
            for path in files[target]:
                viz.drawHeatmap(path)
                viz.drawEdge(path)


if __name__ == "__main__":
    main()
