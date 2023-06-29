# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
from glob import glob
from logging import getLogger


def data_processing(input_dir, out_dir):
    logger = getLogger("res_root").getChild(__name__)

    # パラメータの定義
    items1 = ["density", "enstrophy", "pressure","magfieldx", "magfieldy", "magfieldz", "velocityx","velocityy", "velocityz"]
    items2 = ["magfield1", "magfield2", "magfield3", "velocity1","velocity2", "velocity3"]
    xyz = {1: "x", 2: "y", 3: "z"}

    for target in [4949, 77, 497]:
        logger.debug("Process Start", extra={"addinfo": f"snap{target}"})
        if target == 4949:
            i, j = 49, 49
        elif target == 77:
            i, j = 7, 7
        elif target == 497:
            i, j = 49, 7
        else:
            logger.debug("Error", extra={"addinfo": ""})
            raise "Value Error"

        # bat ファイルの実行
        # 基本的に加工したデータの保存先のフォルダの作成
        logger.debug("MAKE", extra={"addinfo": "ディレクトリの作成"})
        subprocess.run([out_dir + "\\mkdirs.bat", str(target)])

        # ログの保存先
        files = glob(input_dir + "\\*\\ICh.target=50.ares=1.0d-{i}.adiffArt=1.0d-{j}.h00.g00.BCv1=0.0\\Snapshots\\*".format(i=i, j=j))
        for file in files:
            logger.debug("OPEN", extra={"addinfo": f"File {item2}.{'{0:02d}'.format(param)}.{'{0:02d}'.format(job)}"})

            # 元データの分割処理の実行
            subprocess.run([input_dir + "\\..\src\processing\cln\separator.exe", f"{file}"])
            _, _, _, param, job = map(lambda x: int(x) if x.isnumeric() else x,  os.path.basename(file).split("."))

            # 出力されたファイル名の変更
            for item2 in items2:
                if os.path.exists(item2):
                    # ファイル名の変更
                    # magfield1 -> magfieldx
                    os.rename(item2, f"{item2[:-1]}{xyz[int(item2[-1])]}") # separater.exe をもとに分割したファイル名を変換する

                else: # 見つからない場合
                    logger.debug("NotFound", extra={"addinfo": f"ファイル {item2}.{'{0:02d}'.format(param)}.{'{0:02d}'.format(job)} が見つかりませんでした"})

            # 出力されたファイルの移動
            for item1 in items1:
                if os.path.exists(item1):
                    # ファイル名の変更
                    # magfieldx -> magfieldx.01.00
                    newname = f"{item1}.{'{0:02d}'.format(param)}.{'{0:02d}'.format(job)}"
                    os.rename(item1, newname)

                    # ファイルの移動
                    # separater.exe で出力されたファイルは親ディレクトリに生成されるため、逐一移動させる
                    shutil.move(newname, out_dir+f'\\snap{target}\\{item1}\\{"{0:02d}".format(job)}\\')

                else: # 見つからない場合
                    logger.debug("NotFound", extra={"addinfo": f"ファイル {item1}.{'{0:02d}'.format(param)}.{'{0:02d}'.format(job)} が見つかりませんでした"})

            logger.debug("CLOSE", extra={"addinfo": f"File {item2}.{'{0:02d}'.format(param)}.{'{0:02d}'.format(job)}"})


        # coordn を最後に移動させる
        for i in range(1, 4):
            shutil.move("coord" + xyz[i], out_dir+f'\\snap{target}')

        logger.debug("END", extra={"addinfo": "処理終了"})


if __name__ == "__main__":
    import sys
    sys.path.append(".\\")

    from etc.logger import logger_conf
    from params import ROOT_DIR

    logger = logger_conf()
    input_dir = ROOT_DIR + "\\data"
    out_dir = ROOT_DIR + "\\snaps"
    data_processing(input_dir, out_dir)
