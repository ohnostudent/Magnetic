# -*- coding utf-8, LF -*-

import os
import sys
import subprocess
import random
import numpy as np
from glob import glob
from struct import pack
from logging import getLogger
sys.path.append(os.getcwd() + "\src")

from params import SRC_PATH, SNAP_PATH, IMGOUT
from Visualization.SnapData import SnapData

class LicMethod(SnapData):
    logger = getLogger("res_root").getChild(__name__)

    def LIC(self, command):
        self.logger.debug("START", extra={"addinfo": f"execute command {command[0]}"})
        res = subprocess.run(command)
        self.logger.debug("End", extra={"addinfo": f"execute command"})
        return res


    def set_command(self, xfile: str, yfile: str, outname: str, x = False, y = False):
        command = [SRC_PATH + f"\LIC\\LIC.exe", xfile, yfile, outname]
        self.logger.debug("START", extra={"addinfo": f"make command"})
        
        if xfile[-3:] == "npy":
            xdata = np.load(xfile)
            ydata = np.load(yfile)
            command += list(map(str, list(reversed(xdata.shape))))

            # npy1次元のファイルを作って無理やり読み込ます。そして消す。名前はランダムにして
            xtempfile = SRC_PATH + f"\LIC\\xtemp_ohnostrm_reading{random.randint(10000, 99999)}"
            with open(xtempfile, "ab") as f:
                for val in list(xdata.flat)*3: # *3は元のデータがz軸方向に3重なっているのを表現
                    b = pack('f', val)
                    f.write(b)
                f.close()

            ytempfile = SRC_PATH + f"\LIC\ytemp_ohnostrm_reading{random.randint(10000, 99999)}"
            with open(ytempfile, "ab") as f:
                for val in list(ydata.flat)*3: # *3は元のデータがz軸方向に3重なっているのを表現
                    b = pack('f', val)
                    f.write(b)
                f.close()

            command[1], command[2] = xtempfile, ytempfile

        elif x and y:
            command += list(map(str, [x, y]))

        self.logger.debug("COMP", extra={"addinfo": f"make command"})
        return command

    def delete_tempfile(self, xtempfile, ytempfile):
        os.remove(xtempfile)
        os.remove(ytempfile)



def main():
    from src.SetLogger import logger_conf

    logger = logger_conf()
    logger.debug("START", extra={"addinfo": "処理開始"})

    lic = LicMethod()
    datasets  = [77, 497, 4949]
    out_dir = IMGOUT + "\LIC"
    lic.makedir("\LIC")

    for dataset in datasets:
        indir = SNAP_PATH + f"\half\snap{dataset}"
        dir_basename = os.path.basename(indir) # snap77
        base_out_path = out_dir + "\\" + os.path.basename(indir) # .\imgout\LIC\snap77
        lic.makedir(f"\LIC\snap{dataset}")

        binary_paths = glob(indir+"\magfieldx\*\*.npy")
        # ファイルが無い場合
        if binary_paths == []:
            print("Error File not Found")
            return
        
        for xfile in binary_paths[-1:]:
            yfile = xfile.replace("magfieldx", "magfieldy")
            out_path = base_out_path + f"\lic_{dir_basename}.{os.path.splitext(os.path.basename(xfile))[0]}.bmp"
            # print(out_path) # .\imgout\LIC\snap77\lic_snap77.magfieldx.01.14.bmp
            
            command = lic.set_command(xfile, yfile, out_path)
            lic.LIC(command)
            lic.delete_tempfile(xfile, yfile)


if __name__ == "__main__":
    main()
