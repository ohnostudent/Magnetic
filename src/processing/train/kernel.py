# -*- coding utf-8, LF -*-

import numpy as np


class _Kernel:
    """
    visualize.ipynbを参考にkernel作る

        rad = np.arccos(u/np.sqrt(u**2+v**2))
        color2 = np.array(v) / np.array(u)
        color2 = color2 - min(color2.flat)
        color2 = color2/max(color2.flat)
        speed = np.sqrt(u**2 + v**2)
        lw = 7*speed / speed.max()
    ##
    dens = mf.load(mf.gen_snap_path("density",para,job),z=3)
    vX = mf.load(mf.gen_snap_path("velocityX",para,job),z=3)
    vY = mf.load(mf.gen_snap_path("velocityY",para,job),z=3)

    energy = dens * (vX**2 + vY**2) / 2
    ##

    """

    def kernel_listxy(self, im1, im2) -> np.ndarray:  # xy交互のリストを持った行列を返す。shapeが1次元増えるので使わない
        res = np.empty([im1.shape[0], im1.shape[1]])
        res = [[[] for _ in range(im1.shape[1])] for _ in range(im1.shape[0])]
        for x in range(im1.shape[1]):
            for y in range(im1.shape[0]):
                res[y][x] = [im1[y][x], im2[y][x]]
        return np.array(res)

    def kernel_xy(self, im1, im2) -> np.ndarray:  # xy交互の行列を返す。shapeのx方向が2倍になる。
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
