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

    def LIC(self, command: list):
        """
        LIC法可視化の実行
        /IMGOUT/LIC 配下に .bmp を作成

        Args:
            command (list[str])

        Returns:
            result (CompletedProcess) : 処理結果
        """
        result = subprocess.run(command)
        return result

    def set_command(self, xfile: str, yfile: str, outname: str, x = False, y = False) -> list:
        """
        LIC.exe の引数を作成する関数

        Args:
            xfile (str) : 可視化を行う magfieldx のパス
            yfile (str) : 可視化を行う magfieldy のパス
            outname (str) : 出力先のファイル
            x (bool) : わからん, default False
            y (bool) : わからん, default False

        Returns:
            command (list[str]) : .exe 実行用引数の配列

        """

        command = [SRC_PATH + f"/LIC/LIC.exe", xfile, yfile, outname]
        self.logger.debug("START", extra={"addinfo": f"make command\n"})
        
        if (xfile[-4:] == ".npy") and (yfile[-4:] == ".npy"):
            xdata = np.load(xfile)
            ydata = np.load(yfile)
            command += list(map(str, list(reversed(xdata.shape))))

            # npy1次元のファイルを作って無理やり読み込ます。そして消す。名前はランダムにして
            xtempfile = self._create_tempfile(xdata, 'x')
            ytempfile = self._create_tempfile(ydata, 'y')

            command[1], command[2] = xtempfile, ytempfile # 引数に指定

        elif x and y:
            command += list(map(str, [x, y]))
        
        # else:
        #     pass

        self.logger.debug("COMP", extra={"addinfo": f"make command\n"})
        return command

    def _create_tempfile(self, data, xy: str) -> str:
        """
        temp ファイルの作成

        Args:
            data (str) : 読み込んだ numpy データ
            xy (str) : x か y か

        Returns:
            tempfile_path (str) : temp ファイルのパス
        
        """
        tempfile_path = SRC_PATH + f"/LIC/temp/lic_command_{xy}_reading{random.randint(10000, 99999)}.temp"
        with open(tempfile_path, "ab") as f:
            for val in list(data.flat)*3: # *3は元のデータがz軸方向に3重なっているのを表現
                b = pack('f', val)
                f.write(b)
            f.close()
        return tempfile_path

    def delete_tempfile(self, xtempfile: str, ytempfile: str) -> None:
        """
        command 作成時に生成した tempファイルの削除を行う関数

        Args:
            xtempfile (str) : 削除するファイルのパス
            ytempfile (str) : 削除するファイルのパス

        Returns:
            None

        """
        os.remove(xtempfile)
        os.remove(ytempfile)



def mainProcess(logger, lic: LicMethod, dir_basename: str, base_out_path: str, binary_paths: list[str]):
    for xfile in binary_paths:
        try:
            logger.debug("START", extra={"addinfo": f"{os.path.splitext(os.path.basename(xfile))[0]} 開始\n"})
            yfile = xfile.replace("magfieldx", "magfieldy")
            file_name = os.path.splitext(os.path.basename(xfile.replace("magfieldx", "magfield")))
            out_path = base_out_path + f"/lic_{dir_basename}.{file_name[0]}.bmp"
            print(out_path) # ./imgout/LIC/snap77/lic_snap77.magfieldx.01.14.bmp

            if not os.path.exists(out_path):
                # 引数の作成
                command = lic.set_command(xfile, yfile, out_path)
                # 実行 (1画像30分程度)
                # 22:12:46 -> 22:46:12 (34分)
                lic.LIC(command)

                # temp ファイルの削除
                lic.delete_tempfile(command[1], command[2])
            logger.debug("END", extra={"addinfo": f"{os.path.splitext(os.path.basename(xfile))[0]} 終了\n"})

        except KeyboardInterrupt:
            break

        except Exception as e:
            logger.debug(str(e), extra={"addinfo": f"{os.path.splitext(os.path.basename(xfile))[0]} 中断\n"})



def LICMainProcess():
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

    from SetLogger import logger_conf
    from params import datasets
    from concurrent.futures import ThreadPoolExecutor

    try:
        # ログ取得の開始
        logger = logger_conf()
        logger.debug("START", extra={"addinfo": "処理開始\n\n"})

        dataset = int(input("使用するデータセットを入力してください (77/497/4949): "))
        if dataset not in datasets:
            logger.debug("ERROR", extra={"addinfo": "このデータセットは使用できません\n"})
            sys.exit()

        logger.debug("START", extra={"addinfo": f"{dataset} 開始\n"})
        lic = LicMethod()
        
        # 入出力用path の作成
        indir = SNAP_PATH + f"/half/snap{dataset}"
        dir_basename = os.path.basename(indir) # snap77
        out_dir = IMGOUT + "/LIC"
        base_out_path = out_dir + "/" + os.path.basename(indir) # ./imgout/LIC/snap77
        lic.makedir(f"/LIC/snap{dataset}")

        # バイナリファイルの取得
        binary_paths = glob(indir+"/magfieldx/*/*.npy")
    
        # ファイルが無い場合
        if binary_paths == []:
            logger.debug("ERROR", extra={"addinfo": f"File not Found\n"})
            return

        file_count = len(binary_paths)
        with ThreadPoolExecutor(max_workers=6) as exec: # 並列処理 # max_workers は自信のCPUのコア数と相談してください
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[: file_count // 6])
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[file_count // 6 : file_count // 6 * 2])
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[file_count // 6 * 2 : file_count // 6 * 3])
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[file_count // 6 * 3 : file_count // 6 * 4])
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[file_count // 6 * 4 : file_count // 6 * 5])
            exec.submit(mainProcess, logger, lic, dir_basename, base_out_path, binary_paths[file_count // 6 * 5 :])
        
        logger.debug("END", extra={"addinfo": f"{dataset} 終了\n"})

    except KeyboardInterrupt:
        logger.debug("ERROR", extra={"addinfo": f"処理中断\n"})

    except Exception as e:
        logger.debug("ERROR", extra={"addinfo": f"{e}\n"})
    
    finally:
        logger.debug("END", extra={"addinfo": "処理終了"})


if __name__ == "__main__":
    LICMainProcess()

