# -*- coding utf-8, LF -*-

import os
import shutil
import subprocess
import sys
from glob import glob
from logging import getLogger

sys.path.append(os.getcwd())

from config.params import BIN_PATH, ROOT_DIR, SNAP_PATH, SRC_PATH


def set_ij(dataset) -> tuple[int, int] | bool:
    if dataset == 4949:
        i, j = 49, 49
    elif dataset == 77:
        i, j = 7, 7
    elif dataset == 497:
        i, j = 49, 7
    else:
        return False
    return i, j


def move_file(dataset, param, job, item1):
    # ファイル名の変更
    # magfieldx -> magfieldx.01.00
    newname = f"{item1}.{'{0:02d}'.format(param)}.{'{0:02d}'.format(job)}"
    os.rename(item1, newname)

    # ファイルの移動
    # separator.exe で出力されたファイルは親ディレクトリに生成されるため、逐一移動させる
    shutil.move(newname, SNAP_PATH + f'/snap{dataset}/{item1}/{"{0:02d}".format(job)}//')


def rename_file(xyz, item2):
    # ファイル名の変更
    # magfield1 -> magfieldx
    os.rename(item2, f"{item2[:-1]}{xyz[int(item2[-1])]}")  # separator.exe をもとに分割したファイル名を変換する


def dataProcessing() -> None:
    logger = getLogger("res_root").getChild(__name__)

    # パラメータの定義
    items1 = ["density", "enstrophy", "pressure", "magfieldx", "magfieldy", "magfieldz", "velocityx", "velocityy", "velocityz"]
    items2 = ["magfield1", "magfield2", "magfield3", "velocity1", "velocity2", "velocity3"]
    xyz = {1: "x", 2: "y", 3: "z"}

    for dataset in [4949, 77, 497]:
        logger.debug("Process Start", extra={"addinfo": f"snap{dataset}"})

        ij = set_ij(dataset)
        if ij:
            i, j = ij # type: ignore

        else:
            logger.debug("Value Error", extra={"addinfo": "入力したデータセットは使用できません"})
            return

        # bat ファイルの実行
        # 基本的に加工したデータの保存先のフォルダの作成
        logger.debug("MAKE", extra={"addinfo": "ディレクトリの作成"})
        subprocess.run([BIN_PATH + "/Snaps.bat", str(dataset)])

        # ログの保存先
        files = glob(ROOT_DIR + f"/data/ICh.dataset=50.ares=1.0d-{i}.adiffArt=1.0d-{j}.h00.g00.BCv1=0.0/Snapshots/*")
        for file in files:
            # 元データの分割処理の実行
            subprocess.run([SRC_PATH + "/Processing/cln/separator.exe", f"{file}"])
            _, _, _, param, job = map(lambda x: int(x) if x.isnumeric() else x, os.path.basename(file).split("."))
            logger.debug("OPEN", extra={"addinfo": f"File snap{i}{j}.{param:02d}.{job:02d}"})

            # 出力されたファイル名の変更
            for item2 in items2:
                if os.path.exists(item2):
                    rename_file(xyz, item2)

                else:  # 見つからない場合
                    logger.debug("NotFound", extra={"addinfo": f"ファイル {item2}.{param:02d}.{job:02d}"})

            # 出力されたファイルの移動
            for item1 in items1:
                if os.path.exists(item1):
                    move_file(dataset, param, job, item1)

                else:  # 見つからない場合
                    logger.debug("NotFound", extra={"addinfo": f"ファイル {item1}.{param:02d}.{job:02d}"})

            logger.debug("CLOSE", extra={"addinfo": f"File snap{i}{j}.{param:02d}.{job:02d}"})

        # coordn を最後に移動させる
        for i in range(1, 4):
            shutil.move("coord" + xyz[i], SNAP_PATH + f"/snap{dataset}")

        logger.debug("END", extra={"addinfo": "処理終了"})


if __name__ == "__main__":
    from config.SetLogger import logger_conf

    logger = logger_conf()

    dataProcessing()
