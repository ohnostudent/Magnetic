# -*- coding utf-8, LF -*-

import os
import sys
import numpy as np
import pandas as pd
from glob import glob
sys.path.append(os.getcwd() + "/src")

from params import SNAP_PATH, ML_DATA_DIR, datasets, variable_parameters
from Processing.kernel import _kernel


class crateTrain(_kernel):
    res = {0: 'n', 1: 'x', 2: 'o'}

    def __init__(self) -> None:
        pass

    def cut_and_save(self, dataset, val_param) -> None:
        """
        ラベリング時の座標をもとに、元データを切り取る
        """
        df = pd.read_csv(ML_DATA_DIR + f"/LIC_labels/label_snap{dataset}_all.csv", encoding="utf-8")
        df_snap: pd.DataFrame = df[df["dataset"] == dataset]

        for _, d in df_snap.iterrows():
            para, job, side = d["para"], d["job"], d["side"]
            centerx, xlow, xup = d["centerx"], int(d["xlow"]), int(d["xup"])
            centery, ylow, yup = d["centery"], int(d["ylow"]), int(d["yup"])
            label = d["label"]

            img = np.load(SNAP_PATH + f"/half_{side}/snap{dataset}/{val_param}/{job:02d}/{val_param}.{para:02d}.{job:02d}.npy")
            separated_im = img[ylow: yup, xlow: xup]

            base_path = ML_DATA_DIR + f"/snap{dataset}/point_{self.res[label]}/{val_param}"
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            np.save(base_path + f"/{val_param}_{dataset}.{para:02d}.{job:02d}_{centerx}.{centery}", separated_im)


    def fusionnpy2val_param(self, impath, val_param1, val_param2, carnel, OUT_DIR, outbasename):
        im1 = np.load(impath)
        im2 = np.load(impath.replace(val_param1, val_param2))

        resim = carnel(im1,im2)
        
        outpath = OUT_DIR + outbasename + "/" + os.path.basename(impath).replace(val_param1, outbasename)
        np.save(outpath, resim)
        print(outpath)
    

    def fusionnpy3val_param(self, impath, val_param1, val_param2, val_param3, carnel, OUT_DIR, outbasename):
        im1 = np.load(impath)
        im2 = np.load(impath.replace(val_param1, val_param2))
        im3 = np.load(impath.replace(val_param1, val_param3))

        resim = carnel(im1,im2, im3)
        
        outpath = OUT_DIR + outbasename + "/" + os.path.basename(impath).replace(val_param1, outbasename)
        np.save(outpath, resim)
        print(outpath)


def makeFusionData(dataset):
    md = crateTrain()
    return


def makeTrainingData(dataset):
    md = crateTrain()

    val_param_tuples = [(["magfieldx", "magfieldy"], "mag_tupledxy", md.carnellistxy)]
    npys0 = ML_DATA_DIR + f"/LIC_labels/label_snap{dataset}_all.csv"
    npys1X = ML_DATA_DIR + f"/LIC_labels/label_snap{dataset}_all.csv"
    npys1O = ML_DATA_DIR + f"/LIC_labels/label_snap{dataset}_all.csv"

    outdir_ = ML_DATA_DIR
    outdir_x = ML_DATA_DIR
    outdir_o = ML_DATA_DIR

    #/imgout/0131_not/density/density_49.50.8_9.528
    for val_param, outbasename, carnel in val_param_tuples:
        for OUT_DIR, npys in [(outdir_, npys0), (outdir_x, npys1X), (outdir_o, npys1O)]:
            for impath in glob(npys + val_param[0] +"/*"): 
                if  not os.path.exists(OUT_DIR+f"{outbasename}"):
                    os.mkdir(OUT_DIR+f"{outbasename}")

                md.fusionnpy2val_param(impath, *val_param, carnel, OUT_DIR, outbasename)
                

    val_param_tuples = [(["velocityx", "velocityy", "density"], "energy", md.carnelEnergy)]
    #/imgout/0131_not/density/density_49.50.8_9.528
    for val_param, outbasename, carnel in val_param_tuples:
        for OUT_DIR, npys in [(outdir_, npys0), (outdir_x, npys1X), (outdir_o, npys1O)]:
            for impath in glob(npys + val_param[0] +"/*"): 
                if  not os.path.exists(OUT_DIR+f"{outbasename}"):
                    os.mkdir(OUT_DIR+f"{outbasename}")
    
                md.fusionnpy3val_param(impath, *val_param, carnel, OUT_DIR, outbasename)


if __name__ == "__main__":
    makeTrainingData()
