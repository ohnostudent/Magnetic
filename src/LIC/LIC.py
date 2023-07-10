# -*- coding utf-8, LF -*-

import os
import sys
import subprocess
import random
import numpy as np
from glob import glob
from struct import pack
from logging import getLogger
sys.path.append(os.getcwd() + "/src")

from params import SRC_PATH, SNAP_PATH, IMGOUT
from Visualization.SnapData import SnapData


class LicMethod(SnapData):
    logger = getLogger("res_root").getChild(__name__)

    def LIC(self, props: list):
        """
        LIC法可視化の実行
        /IMGOUT/LIC 配下に .bmp を作成

        Args:
            props (list[str])

        Returns:
            result (CompletedProcess) : 処理結果
        """

        # .exe の実行
        result = subprocess.run(props)
        return result

    def set_command(self, xfile: str, yfile: str, outname: str) -> list:
        """
        LIC.exe の引数を作成する関数

        Args:
            xfile (str) : 可視化を行う magfieldx のパス
            yfile (str) : 可視化を行う magfieldy のパス
            outname (str) : 出力先のファイル
            x (bool) : わからん, default False
            y (bool) : わからん, default False

        Returns:
            props (list[str]) : .exe 実行用引数の配列

        """

        self.logger.debug("START", extra={"addinfo": f"make props\n"})

        # コマンドの作成
        props = [SRC_PATH + f"/LIC/LIC.exe", xfile, yfile, outname]
        xfile_isnot_exist = not os.path.exists(xfile)
        yfile_isnot_exist = not os.path.exists(yfile)

        if xfile_isnot_exist and yfile_isnot_exist: # どちらかがない場合
            props += list(map(str, [xfile_isnot_exist, yfile_isnot_exist]))

        elif (xfile[-4:] == ".npy") and (yfile[-4:] == ".npy"):
            # ファイルのロード
            xdata = np.load(xfile)
            ydata = np.load(yfile)
            props += list(map(str, list(reversed(xdata.shape))))

            # npy1次元のファイルを作って無理やり読み込ます。そして消す。名前はランダムにして
            xtempfile = self._create_tempfile(xdata, 'x')
            ytempfile = self._create_tempfile(ydata, 'y')

            props[1], props[2] = xtempfile, ytempfile # 引数に指定
        
        # else:
        #     pass

        self.logger.debug("COMP", extra={"addinfo": f"make props\n"})
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
        while True: # 同じファイルを作成しないようにする
            tempfile_path = SRC_PATH + f"/LIC/temp/lic_command_{xy}_reading{random.randint(10000, 99999)}.temp"

            if not os.path.exists(tempfile_path):
                break

        with open(tempfile_path, "wb") as f: # .tempファイルに書き込み
            for val in list(data.flat)*3: # *3は元のデータがz軸方向に3重なっているのを表現
                b = pack('f', val)
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



def mainProcess(logger, lic: LicMethod, dir_basename: str, base_out_path: str, binary_paths: list[str]):
    for xfile in binary_paths:
        try:
            logger.debug("START", extra={"addinfo": f"{os.path.splitext(os.path.basename(xfile))[0]} 開始\n"})
            yfile = xfile.replace("magfieldx", "magfieldy")
            file_name = os.path.splitext(os.path.basename(xfile.replace("magfieldx", "magfield")))
            out_path = base_out_path + f"/lic_{dir_basename}.{os.path.basename(base_out_path)}.{file_name[0]}.bmp"
            # print(out_path) # ./imgout/LIC/snap77/lic_snap77.magfieldx.01.14.bmp

            if not os.path.exists(out_path):
                # 引数の作成
                props = lic.set_command(xfile, yfile, out_path)
                # 実行 (1画像20分程度)
                lic.LIC(props)

                # temp ファイルの削除
                lic.delete_tempfile(props[1], props[2])

            logger.debug("END", extra={"addinfo": f"{os.path.splitext(os.path.basename(xfile))[0]} 終了\n"})

        except KeyboardInterrupt:
            break

        except Exception as e:
            logger.debug(str(e), extra={"addinfo": f"{os.path.splitext(os.path.basename(xfile))[0]} 中断\n"})


from params import datasets
def LICMainProcess(logger, dataset, size):
    """
    処理時間の目安
    snap77   : 778(ファイル) * 30(分) / 60 / 4 (並列スレッド数) * (CPU速度(GHz) / 2.8(GHz))
    -> 64.833 (時間)

    snap497  : 791(ファイル) * 30(分) / 60 / 4 (並列スレッド数) * (CPU速度(GHz) / 2.8(GHz))
    -> 65.9167 (時間)
    
    snap4949 : 886(ファイル) * 30(分) / 60 / 4 (並列スレッド数) * (CPU速度(GHz) / 2.8(GHz))
    -> 73.83 (時間)

    計     : 2455 * 30(分) / 60 / 並列スレッド数 * (CPU速度(GHz) / 2.8(GHz))
    -> 204.58 (時間)

    """

    from concurrent.futures import ThreadPoolExecutor

    try:
        # ログ取得の開始
        logger.debug("START", extra={"addinfo": "処理開始\n\n"})

        # dataset = int(input("使用するデータセットを入力してください (77/497/4949): "))
        if dataset not in datasets:
            logger.debug("ERROR", extra={"addinfo": "このデータセットは使用できません\n"})
            sys.exit()

        logger.debug("START", extra={"addinfo": f"{dataset}.{size.split('_')[1]} 開始\n"})
        lic = LicMethod()
        
        # 入出力用path の作成
        indir = SNAP_PATH + f"/{size}/snap{dataset}"
        dir_basename = os.path.basename(indir) # snap77
        out_dir = IMGOUT + "/LIC"
        base_out_path = out_dir + "/" + os.path.basename(indir) + "/" + size.split('_')[1] # ./imgout/LIC/snap77/left
        lic.makedir(f"/LIC/snap{dataset}/{size.split('_')[1]}")

        # バイナリファイルの取得
        binary_paths = glob(indir+"/magfieldx/*/*.npy")
    
        # ファイルが無い場合
        if binary_paths == []:
            logger.debug("ERROR", extra={"addinfo": f"File not Found\n"})
            return

        file_count = len(binary_paths)
        with ThreadPoolExecutor() as exec: # 並列処理 # max_workers は自信のCPUのコア数と相談してください
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[: file_count // 10])
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[file_count // 10 : file_count // 10 * 2])
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[file_count // 10 * 2 : file_count // 10 * 3])
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[file_count // 10 * 3 : file_count // 10 * 4])
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[file_count // 10 * 4 : file_count // 10 * 5])
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[file_count // 10 * 5 : file_count // 10 * 6])
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[file_count // 10 * 6 : file_count // 10 * 7])
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[file_count // 10 * 7 : file_count // 10 * 8])
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[file_count // 10 * 8 : file_count // 10 * 9])
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[file_count // 10 * 9 :])
        
        logger.debug("END", extra={"addinfo": f"{dataset} 終了\n"})

    except KeyboardInterrupt:
        logger.debug("ERROR", extra={"addinfo": f"処理中断\n"})

    except Exception as e:
        logger.debug("ERROR", extra={"addinfo": f"{e}\n"})
    
    finally:
        logger.debug("END", extra={"addinfo": "処理終了"})


if __name__ == "__main__":
    from SetLogger import logger_conf

    logger = logger_conf()
    for dataset in datasets:
        for size in ["half_left", "half_right"]:
            LICMainProcess(logger, dataset, size)
    
