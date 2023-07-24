# -*- coding: utf-8 -*-

# 標準モジュールのインポート
import os
import sys
from glob import glob
sys.path.append(os.getcwd() + "\src")

from params import SRC_PATH, IMGOUT, datasets


def _sort_paths(path_list):
    """
    pathlistはglob等で取得したlist。
    pathlistをparam, jobでソートして返す。
    """
    pjp =  list(map(lambda x: list(map(lambda y: int(y) if y.isnumeric() else y, x)),  map(lambda path: [path] + os.path.basename(path).split(".")[2: 4], path_list)))
    # params, job の順にソート
    pjp_sorted = sorted(pjp, key=lambda x: (x[1], x[2]))

    return list(map(lambda x: x[0], pjp_sorted))


def createViewer(dataset):
    from logging import getLogger
    logger = getLogger("res_root").getChild(__name__)

    for size in ["left", "right"]:
        # paths = _sort_paths(paths) # snapの命名規則をもとに時系列順に並び変える。
        paths = glob(IMGOUT + f"/LIC/snap{dataset}/{size}/*.bmp")
        paths_sorted = _sort_paths(paths)

        # viewer用のファイル列を作成する
        pathliststr = "\n"
        for path in paths_sorted:
            path = path.replace("\\", "/")
            pathliststr += f"\t\t\t'{path}', \n"
        
        # html の読み込み
        with open(SRC_PATH + "/Processing/viewer/template/viewer_template.html", 'r', encoding="utf-8") as f:
            html = f.read()
        # 可視化した.bmpのpathの一覧をhtml に追記
        html = html.replace("{ replaceblock }", pathliststr)

        # html の保存
        outname = SRC_PATH + f"/Processing/viewer/template/lic_viewer{dataset}.{size}.html"
        with open(outname, "w", encoding="utf8") as f:
            f.write(html)


if __name__ == '__main__':
    dataset = input("使用するデータセットを入力してください(77/497/4949) : ")

    if dataset.isnumeric(): # 数字以外を入力した場合
        dataset = int(dataset)
    else:
        print("入力内容に間違いがあります")
        sys.exit()

    if dataset not in datasets: # 使用できない場合
        print("不明なデータセットです")
        sys.exit()

    createViewer(dataset)
