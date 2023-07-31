# -*- coding utf-8, LF -*-

import shutil
import os
from glob import glob


def unpacking(root_dir):
    zip_files = glob(root_dir+"\\*.zip") # zipファイルの path の取得
    print("【 処理開始 】\n")

    print("【 解凍開始 ")
    # フォルダがない場合、作成
    # mkdir.cmd の実行でも可
    # その場合は `os.system("mkdir.cmd")` を使用
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    # 各zipファイルの解凍
    for file in zip_files:
        print("【 解凍中 】  : {file}".format(file=file))
        shutil.unpack_archive(file, root_dir)
    
    print("【 解凍終了 】:}")
    print()

    print("【 処理終了 】")


if __name__ == "__main__":
    root_dir = os.getcwd() + "\\snap\\"
    unpacking()
