# -*- coding utf-8, LF -*-

import os
import random
import subprocess
import sys
from glob import glob
from logging import getLogger
from struct import pack

import numpy as np

sys.path.append(os.getcwd() + "/src")

from config.params import DATASETS, IMAGE_PATH, SNAP_PATH, SRC_PATH
from Visualization.Visualize.SnapData import SnapData


class LicMethod(SnapData):
    logger = getLogger("LIC").getChild("LIC")

    def LIC(self, props: list):
        """
        LIC法可視化の実行
        /IMAGE_PATH/LIC 配下に .bmp を作成

        Args:
            props (list[str])

        Returns:
            result (CompletedProcess) : 処理結果
        """

        # .exe の実行
        result = subprocess.run(props, check=True)
        return result

    def set_command(self, xfile: str, yfile: str, out_name: str) -> list:
        """
        LIC.exe の引数を作成する関数

        Args:
            xfile (str) : 可視化を行う magfieldx のパス
            yfile (str) : 可視化を行う magfieldy のパス
            out_name (str) : 出力先のファイル

        Returns:
            props (list[str]) : .exe 実行用引数の配列

        """

        self.logger.debug("START", extra={"addinfo": "make props"})

        # コマンドの作成
        props = [SRC_PATH + "/Visualization/LIC/LIC.exe", xfile, yfile, out_name]
        xfile_is_not_exist = not os.path.exists(xfile)
        yfile_is_not_exist = not os.path.exists(yfile)

        if xfile_is_not_exist and yfile_is_not_exist:  # どちらかがない場合
            props += list(map(str, [xfile_is_not_exist, yfile_is_not_exist]))

        elif (xfile[-4:] == ".npy") and (yfile[-4:] == ".npy"):
            # ファイルのロード
            xdata = np.load(xfile)
            ydata = np.load(yfile)
            props += list(map(str, list(reversed(xdata.shape))))

            # npy1次元のファイルを作って無理やり読み込ます。そして消す。名前はランダムにして
            xtempfile = self._create_tempfile(xdata, "x")
            ytempfile = self._create_tempfile(ydata, "y")

            props[1], props[2] = xtempfile, ytempfile  # 引数に指定

        else:
            self.logger.debug("ERROR", extra={"addinfo": ""})
            raise ValueError

        self.logger.debug("COMP", extra={"addinfo": "make props"})
        return props

    def _create_tempfile(self, data, xy: str) -> str:
        """
        temp ファイルの作成

        Args:
            data (str) : 読み込んだ numpy データ
            xy (str) : x か y か

        Returns:
            tempfile_path (str) : temp ファイルのパス

        """
        # コマンドを保存するための.tempファイルの作成
        while True:  # 同じファイルを作成しないようにする
            tempfile_path = SRC_PATH + f"/Visualization/LIC/temp/lic_command_{xy}_reading{random.randint(10000, 99999)}.temp"

            if not os.path.exists(tempfile_path):
                break

        with open(tempfile_path, "wb") as f:  # .tempファイルに書き込み
            for val in list(data.flat) * 3:  # *3は元のデータがz軸方向に3重なっているのを表現
                b = pack("f", val)
                f.write(b)
            f.close()

        return tempfile_path

    def delete_tempfile(self, xtempfile: str, ytempfile: str) -> None:
        """
        props 作成時に生成した tempファイルの削除を行う関数

        Args:
            xtempfile (str) : 削除するファイルのパス
            ytempfile (str) : 削除するファイルのパス

        Returns:
            None

        """

        # 一時的に作成した .tempファイルの削除
        os.remove(xtempfile)
        os.remove(ytempfile)


def _main_process(lic: LicMethod, dataset: int, base_out_path: str, binary_paths: list[str]) -> None:
    logger = getLogger("LIC").getChild("LIC_main")

    for xfile in binary_paths:
        logger.debug("START", extra={"addinfo": f"{os.path.splitext(os.path.basename(xfile))[0]} 開始"})
        file_name = os.path.splitext(os.path.basename(xfile.replace("magfieldx", "magfield")))
        out_path = base_out_path + f"/lic_snap{dataset}.{os.path.basename(base_out_path)}.{file_name[0]}.bmp"
        # print(out_path) # ./IMAGE_PATH/LIC/snap77/left/lic_snap77.left.magfield.01.14.bmp

        if not os.path.exists(out_path):
            yfile = xfile.replace("magfieldx", "magfieldy")
            props = lic.set_command(xfile, yfile, out_path)
            # 引数の作成
            # 実行 (1画像20分程度)
            lic.LIC(props)

            # temp ファイルの削除
            lic.delete_tempfile(props[1], props[2])

        logger.debug("END", extra={"addinfo": f"{os.path.splitext(os.path.basename(xfile))[0]} 終了"})


def _main(dataset: int, side: str) -> None:
    """
    処理時間の目安
    1ファイル : 20(分) (3.98(GHz))

    snap77   : 778(ファイル) * 20(分) / 60 / 4 (並列スレッド数) * (CPU速度(GHz) / 3.98(GHz))
    -> 64.833 (時間)

    snap497  : 791(ファイル) * 20(分) / 60 / 4 (並列スレッド数) * (CPU速度(GHz) / 3.98(GHz))
    -> 65.9167 (時間)

    snap4949 : 886(ファイル) * 20(分) / 60 / 4 (並列スレッド数) * (CPU速度(GHz) / 3.98(GHz))
    -> 73.83 (時間)

    計     : 2455(ファイル) * 20(分) / 60 / 並列スレッド数 * (CPU速度(GHz) / 3.98(GHz))
    -> 204.58 (時間)
    """
    from concurrent.futures import ThreadPoolExecutor

    logger = getLogger("LIC").getChild("LIC_main")

    if dataset not in DATASETS:
        logger.debug("ERROR", extra={"addinfo": "このデータセットは使用できません"})
        sys.exit()

    logger.debug("START", extra={"addinfo": f"{dataset}.{side.split('_')[1]} 開始"})

    if not os.path.exists(SRC_PATH + "/Visualization/LIC/LIC.exe"):
        raise FileNotFoundError

    lic = LicMethod()

    # 入出力用path の作成
    base_out_path = IMAGE_PATH + f"/LIC/snap{dataset}/{side.split('_')[1]}"  # ./images/LIC/snap77/left
    lic.makedir(f"/LIC/snap{dataset}/{side.split('_')[1]}")

    # バイナリファイルの取得
    binary_paths = glob(SNAP_PATH + f"/{side}/snap{dataset}/magfieldx/*/*.npy")
    file_count = len(binary_paths)

    # ファイルが無い場合
    if file_count == 0:
        logger.debug("ERROR", extra={"addinfo": "File not Found"})
        return

    else:
        logger.debug("FILE COUNT", extra={"addinfo": f"{file_count}"})

    with ThreadPoolExecutor() as exec:  # 並列処理 # max_workers は自信のCPUのコア数と相談してください
        exec.submit(_main_process, lic, dataset, base_out_path, binary_paths[: file_count // 5])
        exec.submit(_main_process, lic, dataset, base_out_path, binary_paths[file_count // 5 : file_count // 5 * 2])
        exec.submit(_main_process, lic, dataset, base_out_path, binary_paths[file_count // 5 * 2 : file_count // 5 * 3])
        exec.submit(_main_process, lic, dataset, base_out_path, binary_paths[file_count // 5 * 3 : file_count // 5 * 4])
        exec.submit(_main_process, lic, dataset, base_out_path, binary_paths[file_count // 5 * 4 :])

    logger.debug("END", extra={"addinfo": f"{dataset} 終了"})


if __name__ == "__main__":
    from config.SetLogger import logger_conf

    logger = logger_conf("LIC")
    # ログ取得の開始
    logger.debug("START", extra={"addinfo": "処理開始"})

    for dataset in DATASETS:
        for side in ["half_left", "half_right"]:
            _main(dataset, side)

    logger.debug("END", extra={"addinfo": "処理終了"})
