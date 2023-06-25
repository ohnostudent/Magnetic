import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import subprocess
from glob import glob
from dotenv import load_dotenv
# load_dotenv()

class SnapData():
    ROOT_DIR = os.getcwd()
    SNANP_PATH = ROOT_DIR + "\\snaps"
    imgout = ROOT_DIR + "\\imgout\\visualize"

    def __init__(self) -> None:
        if not os.path.exists(self.ROOT_DIR+"\\heatmap"):
            os.makedirs(self.ROOT_DIR+"\\heatmap")

        if not os.path.exists(self.ROOT_DIR+"\\edge"):
            os.makedirs(self.ROOT_DIR+"\\edge")

        if not os.path.exists(self.ROOT_DIR+"\\heatmap"):
            os.makedirs(self.ROOT_DIR+"\\heatmap")
    
    def setSnapData(self, file_path, z=3):
        # データのインポート
        # r : 読み込み, b : バイナリモード
        # if os.path.splitext(file_path)[1] == "npy": # 拡張子の判定
        if file_path[-3:] == "npy": # 拡張子なしの場合を考慮するとこの形になる？
            self.target, self.param, self.job, _ = os.path.basename(file_path).split(".")
            return np.load(file_path)

        self.target, self.param, self.job = os.path.basename(file_path).split(".")        
        with open(file_path, mode="rb") as f:
            if z == 1:
                self.snap_data = np.fromfile(f, dtype='f', sep='').reshape(1025, 513)

            elif z == 3:
                self.snap_data = np.fromfile(f, dtype='f', sep='')[:525825].reshape(1025, 513)

            f.close()
        return None


class Visualize(SnapData):
    ROOT_DIR = os.getcwd()
    SNANP_PATH = ROOT_DIR + "\\snaps"
    imgout = ROOT_DIR + "\\imgout\\visualize"

    def __init__(self) -> None:
        super().__init__(self)

    def drawHeatmap(self, saveimg=False, bar_range=None):
        # フォルダの作成
        if not os.path.exists(self.imgout + f"\\heatmap\\{self.target}\\{self.job :02d}"):
            os.makedirs(self.imgout + f"\\heatmap\\{self.target}\\{self.job :02d}")
            # print(self.imgout + f"\\heatmap\\{self.target}\\{self.job :02d}")

        # 描画
        if bar_range:
            sns.heatmap(self.snap_data, vmin=bar_range[0], vmax=bar_range[1])
        else:
            sns.heatmap(self.snap_data)
        # 保存
        if saveimg:
            plt.savefig(self.imgout + f"\\heatmaps\\{self.target}\\{self.job :02d}\\{self.target}.{self.param :02d}.{self.job :02d}.png")

        # メモリの開放
        # plt.clf()
        # plt.close() 
    

    def drawEdge(self):
        #cv2で扱える0-255の整数に整形
        data = self.snap_data.copy()
        data = (data - min(data.flat))*254/max(data.flat)
        data = data.astype("uint8")
        # print(data)

        edges = cv2.Canny(data, threshold1=150, threshold2=200)
        plt.imshow(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
        cv2.imwrite(self.imgout + f"\\edges\\{self.target}\\{self.job :02d}\\{self.target}.{self.param :02d}.{self.job :02d}.png", edges)
    

class visualize(SnapData):
    def __init__(self) -> None:
        super().__init__()
    
    

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
            carnel1 = mf.ave_carnel(compress)
            carnel2 = carnel1.T
            dataX = mf.convolute(dataX, carnel2,stride=compress)
            dataY = mf.convolute(dataY, carnel1,stride=compress)

        x = range(dataX.shape[1])
        y = range(dataY.shape[0])
        #X,Y方向それぞれのベクトルに対して座標の行列を設定
        X, Y = np.meshgrid(x, y)
        #X,Y方向それぞれのベクトルの強さ
        u = dataX
        v = dataY
        #########rotの計算途中の微分でデータの端っこが削れる
        rot = mf.rot2d(u, v)
        u = u[2:-2,2:-2]
        v = v[2:-2,2:-2]
        X = X[2:-2,2:-2]
        Y = Y[2:-2,2:-2]
        ##########
        color = u**2 + v**2
        color = color*2/max(color.flat)
        rad = np.arccos(u/np.sqrt(u**2+v**2))
        color2 = np.array(v) / np.array(u)
        color2 = color2 - min(color2.flat)
        color2 = color2/max(color2.flat)
        speed = np.sqrt(u**2 + v**2)
        lw = 7*speed / speed.max()
        
        fig = plt.figure(1)
        # plt.contour(X,Y,rad)
        # mf.show(rot,bar_range=[-0.05,0.05])
        # sns.heatmap(dataY)
        # strm = plt.streamplot(X, Y, u, v, density=[5], color=color, arrowstyle='-', linewidth=1,cmap="rainbow")
        strm = plt.streamplot(X, Y, u, v, density=[3], color=rot, arrowstyle='-', linewidth=lw,cmap="rainbow")
        # strm = plt.streamplot(X, Y, u, v, density=[0.5], color=rad, arrowstyle='-', linewidth=1,cmap="rainbow", minlength=0.001)
        rad2 = abs(rad - (3.1415927/2))
        # sns.heatmap(rad2, cmap="bone")
        # strm = plt.streamplot(X, Y, u, v, density=[1,5], color=black, arrowstyle='-|>', linewidth=1)
        
        #fig.colorbar(strm.lines)
        # plt.savefig(f"{imgout}1111/{number}.png")
        # plt.show()
    
    def drawStreamHeatmap(self, magx, magy):
        # magx = glob(self.ROOT_DIR+"snaps/snap4949/magfieldx/*/*")
        # magy = glob(self.ROOT_DIR+"snaps/snap4949/magfieldy/*/*")
        # if len(magx) != len(magy):
        #     print("not much length")

        # print(magx[0])
        # for i in range(760,len(magx)-1,10):
        for i in [len(magx)-2]:
            dataX1 = mf.load(magx[i], z=3)[350:700, 270:590]
            dataX2 = mf.load(magx[i+1], z=3)[350:700, 270:590]
            dataY1 = mf.load(magy[i], z=3)[350:700, 270:590]
            dataY2 = mf.load(magy[i+1], z=3)[350:700, 270:590]
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
            # rot1 = mf.rot2d(dataX1, dataY1)
            # rot2 = mf.rot2d(dataX2, dataY2)
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
            
            #edge
            # data = ((std - min(std.flat))*254/max(std.flat))
            # data = data.astype("uint8")
            # edges = cv2.Canny(data, threshold1=250, threshold2=250)
            # plt.imshow(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
            
            ax1 = plt.subplot(2,1,1)
            ax1.pcolor(rad, cmap="brg", vmax=0.1, vmin=-0.1)
            ax2 =plt.subplot(2,1,2)
            ax2.pcolor(dataX1, vmax = 0.03)            

            #plt.savefig(f"{imgout}1114/{i}_magfield_paraline{os.path.basename(magx[i])[10:]}.png")



def main():
    from glob import glob

    viz = Visualize()
    for target in ["density"]:#["velocityy", "magfieldy", "density", "enstrophy"]:
        files = glob(os.getcwd() + f"\\snaps\\snap77\\{target}\\*\\*")
        for file in files:
            viz.setSnapData(file)
            viz.drawHeatmap(saveimg=True) # Heatmap
            viz.drawEdge()

