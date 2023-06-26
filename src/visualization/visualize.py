import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import subprocess
from glob import glob
from dotenv import load_dotenv
# load_dotenv()

from SnapData import SnapData
from params import ROOT_DIR, IMGOUT, SNANP_PATH


class Visualize(SnapData):
    def __init__(self) -> None:
        super().__init__(self)
    
    def drawEdge(self):
        #cv2で扱える0-255の整数に整形
        data = self.snap_data.copy()
        data = (data - min(data.flat))*254 / max(data.flat)
        data = data.astype("uint8")
        # print(data)

        edges = cv2.Canny(data, threshold1=150, threshold2=200)
        plt.imshow(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
        cv2.imwrite(IMGOUT + f"\\edges\\{self.target}\\{self.job :02d}\\{self.target}.{self.param :02d}.{self.job :02d}.png", edges)
    

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
            carnel1 = self.ave_carnel(compress)
            carnel2 = carnel1.T
            dataX = self.convolute(dataX, carnel2, stride=compress)
            dataY = self.convolute(dataY, carnel1, stride=compress)

        x = range(dataX.shape[1])
        y = range(dataY.shape[0])
        #X,Y方向それぞれのベクトルに対して座標の行列を設定
        X, Y = np.meshgrid(x, y)
        #X,Y方向それぞれのベクトルの強さ
        u = dataX
        v = dataY
        #########rotの計算途中の微分でデータの端っこが削れる
        rot = self.rot2d(u, v)
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
        # plt.savefig(IMGOUT}1111/{number}.png")
        # plt.show()
    

    #畳み込み
    def convolute(self, data: np.array, carnel: np.array, padding=0, stride=1):
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
        # print("result   Y,X:",result_height,result_width)
        # print("original Y,X:",data.shape)

        def loop_calc():
            def calc(array):
                result = sum(array*carnel)
                result = sum(result.flat)
                return result
            
            orgX = 0
            orgY = 0
            convoluted = np.zeros((result_height, result_width))

            for resultY in range(result_height):
                for resultX in range(result_width):
                    array = data[orgY : orgY + c_height, orgX : orgX + c_width]
                    # a = convoluted[resultY]
                    convoluted[resultY][resultX] = calc(array)
                    orgX += stride

                orgX = 0
                orgY += stride
            return convoluted

        return loop_calc()

    def ave_carnel(self, size:int):
        """
        畳み込みにおける平滑化のカーネル作成
        """
        ones = np.ones((size,size))
        res = ones / (size**2)
        return res

    #離散データの微分
    def diff(self, x, h):
        """
        精度低めの普通の微分。誤差:h**2
        """
        res = x[2:] - x[:-2]
        # print(x[1:])
        # print(x[:-1])
        return res/(2*h)
    
    def diff4(self, x, h):
        """
        精度高めの微分・。誤差:h**4
        1回微分{-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)}/12h
        xは時系列データ,hはデータ間の時間(second)
        ベクトル長が4短くなる
        """
        res = -x[4:] + 8*x[3:-1] - 8*x[1:-3] + x[:-4]
        return res/(12*h)
    
    def diff4_xy(self, data: np.ndarray, h:float, vectol: str):
        """
        diff4を使った行列の横方向偏微分
        """
        if vectol == "y":
            data = data.T

        res = np.ndarray([])
        for vec in data:
            if res.shape == tuple():
                res = self.diff4(vec, h)
            else:
                res = np.append(res, self.diff4(vec, h))
        
        if vectol == "y":
            return res.reshape(data.shape[0],data.shape[1]-4).T
        return res.reshape(data.shape[0],data.shape[1]-4)

    def rot2d(self, vX: np.ndarray, vY: np.ndarray):
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
        return self.diff4_xy(vY,1)[2:-2,] - self.diff4_xy(vX,1)[:,2:-2]





class Viz(SnapData):
    def __init__(self) -> None:
        super().__init__()
    
    def drawEnergy_for_velocity(self, dens, vx, vy):
        # 速さと密度について
        energy = dens * (vx**2 + vy**2) / 2
        sns.heatmap(energy)
    
    def drawEnergy_for_magfield(self, magx, magy):
        # 磁場について
        bhoge = (magx**2 + magy**2)/2
        sns.heatmap(bhoge)

    def drawStreamHeatmap(self, magx1, magy1, magx2, magy2):
        dataX1 = magx1[350:700, 270:590]
        dataX2 = magx2[350:700, 270:590]
        dataY1 = magy1[350:700, 270:590]
        dataY2 = magy2[350:700, 270:590]
        dataX = dataX2 - dataX1
        dataY = dataY2 - dataY1

        x = range(dataX.shape[1])
        y = range(dataY.shape[0])
        #X,Y方向それぞれのベクトルに対して座標の行列を設定
        X, Y = np.meshgrid(x, y)
        #X,Y方向それぞれのベクトルの強さ
        u = dataX
        v = dataY
        #########rotの計算途中の微分でデータの端っこが削れる
        # rot1 = rot2d(dataX1, dataY1)
        # rot2 = rot2d(dataX2, dataY2)
        # rot = rot2 - rot1
        # u = u[2:-2,2:-2]
        # v = v[2:-2,2:-2]
        # X = X[2:-2,2:-2]
        # Y = Y[2:-2,2:-2]
        ##########
        def calc_rad(u,v):
            return np.arccos(u/np.sqrt(u**2+v**2))
        
        rad1 = calc_rad(dataX1,dataY1)
        rad2 = calc_rad(dataX2,dataY2)
        rad = rad2-rad1
        reg_rad = abs(rad)
        std = reg_rad
        
        # edge
        # data = ((std - min(std.flat))*254/max(std.flat))
        # data = data.astype("uint8")
        # edges = cv2.Canny(data, threshold1=250, threshold2=250)
        # plt.imshow(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
        
        ax1 = plt.subplot(2,1,1)
        ax1.pcolor(rad, cmap="brg", vmax=0.1, vmin=-0.1)
        ax2 = plt.subplot(2,1,2)
        ax2.pcolor(dataX1, vmax = 0.03)            

        #plt.savefig(IMGOUT}1114/{i}_magfield_paraline{os.path.basename(magx[i])[10:]}.png")



def main():
    from glob import glob

    viz = Visualize()
    for target in ["density"]:#["velocityy", "magfieldy", "density", "enstrophy"]:
        files = glob(ROOT_DIR + f"\\snaps\\snap77\\{target}\\*\\*")
        for file in files:
            viz.setSnapData(file)
            viz.drawHeatmap(saveimg=True) # Heatmap
            viz.drawEdge()

