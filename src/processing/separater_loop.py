# -*- coding: utf-8 -*-

import os
import shutil
import subprocess
from glob import glob


def data_processing(input_dir, out_dir):
    items1 = ["density", "enstrophy", "pressure","magfieldx", "magfieldy", "magfieldz", "velocityx","velocityy", "velocityz"]
    items2 = ["magfield1", "magfield2", "magfield3", "velocity1","velocity2", "velocity3"]
    xyz = {1: "x", 2: "y", 3: "z"}

    for target in [4949, 77, 497]:
        if target == 4949:
            i, j = 49, 49
        elif target == 77:
            i, j = 7, 7
        elif target == 497:
            i, j = 49, 7
        else:
            raise "Value Error"

        # bat ファイルの実行
        # 基本的に加工したデータの保存先のフォルダの作成
        subprocess.run([out_dir + "\\mkdirs.bat", str(target)])
        files = glob(input_dir + "\\*\\ICh.target=50.ares=1.0d-{i}.adiffArt=1.0d-{j}.h00.g00.BCv1=0.0\\Snapshots\\*".format(i=i, j=j))

        # ログの保存先
        f = open(out_dir + f'\\snap{target}\\myfile.txt', 'w')
        f.write(f"[start] snap{target}")

        for file in files:
            # 元データの分割処理の実行
            subprocess.run([input_dir + "\\..\src\processing\cln\separator.exe", f"{file}"])
            _, _, _, param, job = map(lambda x: int(x) if x.isnumeric() else x,  os.path.basename(file).split("."))

            for item2 in items2:
                if os.path.exists(item2):
                    # ファイル名の変更
                    # magfield1 -> magfieldx
                    os.rename(item2, f"{item2[:-1]}{xyz[int(item2[-1])]}") # separater.exe をもとに分割したファイル名を変換する
                else:
                    f.write(f"Filenot Found: {item2}.{'{0:02d}'.format(param)}.{'{0:02d}'.format(job)}\n")
            f.write("\n")

            for item1 in items1:
                if os.path.exists(item1):
                    # ファイル名の変更
                    # magfieldx -> magfieldx.01.00
                    newname = f"{item1}.{'{0:02d}'.format(param)}.{'{0:02d}'.format(job)}"
                    os.rename(item1, newname)

                    # ファイルの移動
                    # separater.exe で出力されたファイルは親ディレクトリに生成されるため、逐一移動させる
                    shutil.move(newname, out_dir+f'\\snap{target}\\{item1}\\{"{0:02d}".format(job)}\\')

                else:
                    f.write(f"Filenot Found: {item2}.{'{0:02d}'.format(param)}.{'{0:02d}'.format(job)}\n")

        # coordn を最後に移動させる
        for i in range(1, 4):
            shutil.move("coord" + xyz[i], out_dir+f'\\snap{target}')

        f.write("[end]")
        f.close()


if __name__ == "__main__":
    import sys
    sys.path.append(".\\src")

    from params import ROOT_DIR
    
    input_dir = ROOT_DIR + "\\data"
    out_dir = ROOT_DIR + "\\snaps"
    data_processing(input_dir, out_dir)
