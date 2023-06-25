import os
import subprocess
import random
# import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
# from PIL import Image
from struct import pack
from dotenv import load_dotenv



def ohno_lic(root_dir, xfile: str, yfile: str, outname: str, x = False, y = False):
    command = [f"{root_dir}\\src\\LIC\\LIC.exe", xfile, yfile, outname]

    if xfile[-3:] == "npy":
        xdata = np.load(xfile)
        ydata = np.load(yfile)
        command += list(map(str, list(reversed(xdata.shape))))

        #npy1次元のファイルを作って無理やり読み込ます。そして消す。名前はランダムにして
        xtempfile = f"xtemp_ohnostrm_reading{random.randint(10000, 99999)}"
        with open(xtempfile, "ab") as f:
            for val in list(xdata.flat)*3:#*3は元のデータがz軸方向に3重なっているのを表現
                b = pack('f', val)
                f.write(b)
            f.close()

        ytempfile = f"ytemp_ohnostrm_reading{random.randint(10000, 99999)}"
        with open(ytempfile, "ab") as f:
            for val in list(ydata.flat)*3:#*3は元のデータがz軸方向に3重なっているのを表現
                b = pack('f', val)
                f.write(b)
            f.close()

        command[1], command[2] = xtempfile, ytempfile
        os.remove(xtempfile)
        os.remove(ytempfile)

    elif (x and y) == True:
        command += list(map(str, [x, y]))

    res = subprocess.run(command)
    print(command)
    return res


def LIC():
    load_dotenv(".env")
    root_dir = os.getcwd()
    ipt_dir = root_dir + "\\snaps"
    indirs  = [ipt_dir+"\\half\\snap77", ipt_dir+"\\half\\snap49", ipt_dir+"\\half\\snap497"]
    out_dir = root_dir + "\\imgout\\ohnolic"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #snap49split1\\magfieldx\\1\\
    for indir in indirs[-1:]:
        basename = os.path.basename(os.path.dirname(indir))
        newoutdir = out_dir+os.path.basename(os.path.dirname(indir))+"\\" #"..\\imgout\\ohnolic\\snap77\\"
        if not os.path.exists(newoutdir):
            os.mkdir(newoutdir)

        binarypaths = glob(indir+"\\magfieldx\\*\\*.npy")
        for xfile in binarypaths[-1:]:
            yfile = xfile.replace("magfieldx", "magfieldy")
            out = f"{newoutdir}lic_{basename}.{xfile[-9:-4]}.bmp"
            
            if not os.path.exists(out):
                ohno_lic(xfile, yfile, out)
            print(out)

if __name__ == "__main__":
    LIC()
