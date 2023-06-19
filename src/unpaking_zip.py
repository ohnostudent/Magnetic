# -*- encoding utf-8, LF -*-

import shutil
import os
from glob import glob

root_dir = os.getcwd() + "\\snap\\"
zip_files = glob(root_dir+"\\org_zip\\*.zip") # zipファイルの path の取得
snap_items = ["snap49", "snap77", "snap497"]


print("【 処理開始 】\n")

for snap in snap_items:
    print("【 解凍開始 】: {snap}".format(snap=snap))
    # フォルダがない場合、作成
    # mkdir.cmd の実行でも可
    # その場合は `os.system("mkdir.cmd")` を使用
    if not os.path.exists(root_dir + snap):
        os.mkdir(root_dir + snap)

    # 各zipファイルの解凍
    for file in zip_files:
        print("【 解凍中 】  : {file}".format(file=file))
        shutil.unpack_archive(file, root_dir + snap)
    
    print("【 解凍終了 】: {snap}".format(snap=snap))
    print()

print("【 処理終了 】")
