# -*- coding: utf-8 -*-

# 標準モジュールのインポート
import os
import sys
from glob import glob

sys.path.append(os.getcwd() + "/src")

from config.params import IMAGE_PATH, SIDES, SRC_PATH


def _sort_paths(path_list: list[str]) -> list[str]:
    """ファイルパスのソート

    入力されたリストを param, job の順でソートして返す

    Args:
        path_list (list[str]): 並び変える前のリスト

    Returns:
        list[str]: 並び変えたリスト
    """
    pjp: list = list(
        map(
            lambda x: list(map(lambda y: int(y) if y.isnumeric() else y, x)),
            map(lambda path: [path] + os.path.basename(path).split(".")[2:4], path_list),
        )
    )
    # params, job の順にソート
    pjp_sorted = sorted(pjp, key=lambda x: (x[1], x[2]))

    return list(map(lambda x: x[0], pjp_sorted))


def createViewer(dataset: int) -> None:
    """viewerを作る関数

    template.html に必要な情報を書き込む

    Args:
        dataset (int): 対象のデータセット
    """
    for size in SIDES:
        # paths = _sort_paths(paths) # snapの命名規則をもとに時系列順に並び変える。
        paths = glob(IMAGE_PATH + f"/LIC/snap{dataset}/{size}/*.bmp")
        paths_sorted = _sort_paths(paths)

        # viewer用のファイル列を作成する
        path_list_str = "\n"
        for path in paths_sorted:
            path_str = path.replace("\\", "/")
            path_list_str += f"\t\t\t'{path_str}', \n"

        # html の読み込み
        with open(SRC_PATH + "/Processing/viewer/template/viewer_template.html", "r", encoding="utf-8") as f:
            html = f.read()

        # 可視化した.bmpのpathの一覧をhtml に追記
        html = html.replace("{ replaceblock }", path_list_str)

        # html の保存
        out_name = SRC_PATH + f"/Processing/viewer/template/lic_viewer{dataset}.{size}.html"
        with open(out_name, "w", encoding="utf8") as f:
            f.write(html)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print()
        sys.exit()
    else:
        dataset_str = sys.argv[1]

    from config.params import set_dataset

    dataset = set_dataset(dataset_str)
    createViewer(dataset)
