# -*- coding: utf-8 -*-

# 標準モジュールのインポート
import os
import sys
from glob import glob
sys.path.append(os.getcwd() + "\src")

from params import SRC_PATH, IMGOUT, datasets


def _sort_paths(pathlist):
    """
    pathlistはglob等で取得したlist。
    locはそれぞれのpath上での位置。
    pathlistをpara, jobでソートして返す。
    """
    pjp =  list(map(lambda x: list(map(lambda y: int(y) if y.isnumeric() else y, x)),  map(lambda path: [path] + os.path.basename(path).split(".")[2:4], pathlist)))
    pjp_sorted = sorted(pjp, key=lambda x: (x[1], x[2]))

    return list(map(lambda x: x[0], pjp_sorted))


def createViewer(dataset):
    # paths = _sort_paths(paths) # snapの命名規則をもとに時系列順に並び変える。
    paths = glob(IMGOUT + f"/LIC/snap{dataset}/*.bmp")
    paths_sorted = _sort_paths(paths)

    # viewer用のファイル列を作成する
    pathliststr = "\n"
    for path in paths_sorted:
        path = path.replace("\\", "/")
        pathliststr += f"\t\t\t'{path}', \n"

    with open(SRC_PATH + "/Processing/viewer/template/viewer_template.html", 'r', encoding="utf-8") as f:
        html = f.read()
    html = html.replace("{ replaceblock }", pathliststr)

    outname = SRC_PATH + f"/Processing/viewer/template/lic_viewer{dataset}.html"
    with open(outname, "w", encoding="utf8") as f:
        f.write(html)


if __name__ == '__main__':
    for dataset in datasets:
        createViewer(dataset)