# -*- coding utf-8, LF -*-

import json
import os
import sys
from logging import getLogger
from math import floor

import pandas as pd

sys.path.append(os.getcwd() + "/src")

from config.params import DATASETS, IMAGE_SHAPE, LABELS, ML_DATA_DIR, SIDES


def _create_json(file_name) -> None:
    """基盤の作成を行う関数
    Args:
        file_name (str): 保存するファイルの名前
    """
    json_dict = _set_default()
    # 保存
    with open(ML_DATA_DIR + f"/LIC_labels/{file_name}.json", "w", encoding="utf-8") as f:
        json.dump(json_dict, f)


def _set_default() -> dict:
    # 基盤の作成
    json_dict = dict()
    for dataset in DATASETS:
        json_dict[dataset] = dict()

        for side in SIDES:
            json_dict[dataset][side] = dict()

            for label in LABELS:
                json_dict[dataset][side][label] = dict()
    return json_dict


def _df_to_dict(df: pd.DataFrame) -> dict:
    """DataFrame を json に変換する関数

    return
    save_list[0] == {
        "7": {
            "x_range": [13, 38],
            "y_center": {"1": 287, "2": 289, "3": 294, "4": 298, "5": 308, "6": 315, "7": 319, "8": 316},
            "y_range": {"1": [262, 312], "2": [264, 314], "3": [244, 344], "4": [248, 348], "5": [258, 358], "6": [265, 365], "7": [269, 369], "8": [266, 366]},
            "shape": {"1": [25, 50], "2": [25, 50], "3": [25, 100], "4": [25, 100], "5": [25, 100], "6": [25, 100], "7": [25, 100], "8": [25, 100]},
            "x_center": 26
        }
    }
    """
    job_list = df["job"].unique()
    result_dict = dict()

    for job in job_list:
        shapes_list = df[df["job"] == job].reset_index(drop=True).to_dict()

        save_dict = dict()
        save_dict["center"] = dict()
        save_dict["x_range"] = dict()
        save_dict["y_range"] = dict()
        save_dict["shape"] = dict()

        for cnt in range(len(shapes_list["centerx"])):
            save_dict["center"][cnt] = [shapes_list["centerx"][cnt], shapes_list["centery"][cnt]]
            save_dict["x_range"][cnt] = [shapes_list["xlow"][cnt], shapes_list["xup"][cnt]]
            save_dict["y_range"][cnt] = [shapes_list["ylow"][cnt], shapes_list["yup"][cnt]]
            save_dict["shape"][cnt] = [shapes_list["width"][cnt], shapes_list["height"][cnt]]

        result_dict[f"{int(job) :02d}"] = save_dict

    return result_dict


def _set_n() -> dict:
    # 切り取る長方形の重心座標のリスト
    # 4 * 2 = 8 点取る
    x_locs = [30, 60, 90, 120, 150, 180, 210]
    y_locs = [150, 450]
    shapes = IMAGE_SHAPE

    save_dict = dict()  # 保存用の辞書
    save_dict["center"] = dict()
    save_dict["x_range"] = dict()
    save_dict["y_range"] = dict()
    save_dict["shape"] = dict()

    for x_idx in range(len(x_locs)):
        for y_idx in range(len(y_locs)):
            k = x_idx * 2 + y_idx
            save_dict["center"][k] = [x_locs[x_idx], y_locs[y_idx]]
            save_dict["x_range"][k] = [x_locs[x_idx] - shapes[0] // 2 - 1, x_locs[x_idx] + shapes[0] // 2]
            save_dict["y_range"][k] = [y_locs[y_idx] - shapes[1] // 2, y_locs[y_idx] + shapes[1] // 2]
            save_dict["shape"][k] = shapes

    return {f"{k :02d}": save_dict for k in range(7, 15)}


def _set_xo(dataset: int, side: str, label: int) -> dict:
    """x点, o点の切り取りに関する関数

    writer.py を用いて切り取ったx点, o点のcsvデータを jsonに変換する

    Args:
        dataset (int): 77 or 497 ot 4949
        side (str): left or right
        label (int): 1(x点), 2(o点)

    Returns:
        dict : 切り取り点に関するjson
    """
    # データのロード
    df_snap = pd.read_csv(ML_DATA_DIR + f"/LIC_labels/label_snap{dataset}_org.csv")
    df_snap = df_snap.sort_values(["side", "label", "para", "job", "centerx"]).reset_index(drop=True)

    # 加工
    df_snap = df_snap[(df_snap["dataset"] == dataset) & (df_snap["side"] == side) & (df_snap["label"] == label)]
    df_snap["xlow"] = df_snap["xlow"].map(lambda x: floor(x))  # 小数の切り上げ
    df_snap = df_snap[df_snap["job"] >= 7]  # 6 以下は使わない

    # サイズの計算
    df_snap["width"] = df_snap["xup"] - df_snap["xlow"]  # 幅
    df_snap["height"] = df_snap["yup"] - df_snap["ylow"]  # 高さ

    # パラメータ毎にデータを整形
    columns = ["dataset", "para", "job", "centerx", "centery", "xlow", "xup", "ylow", "yup", "width", "height", "label"]
    df_snap_list = [df_snap[df_snap["para"] == p][columns].reset_index(drop=True) for p in df_snap["para"].unique()]
    del df_snap

    # 要約の作成
    # job, 位置固定、param 変動
    df_all = pd.concat(df_snap_list)  # 全結合
    columns = ["para", "job", "centerx", "centery", "xlow", "xup", "ylow", "yup", "width", "height"]
    df_describe_all = list()
    for i in range(len(df_snap_list[0])):
        df_b: pd.DataFrame = df_all[df_all.index == i][columns]
        df_describe_all.append(df_b.describe().round(3))
    del df_all

    # 中央値(50%)を基に基準値を作成
    df_median = pd.DataFrame(columns=columns)

    for df_describe_i in df_describe_all:
        df_median = pd.concat([df_median, df_describe_i.loc[["50%"], columns]])

    df_median = df_median.reset_index(drop=True).reset_index().astype(int)
    df_median["para"] = df_median["index"]
    df_median["width"] = df_median["xup"] - df_median["xlow"]
    df_median["height"] = df_median["yup"] - df_median["ylow"]
    df_median = df_median.set_index("index")
    df_median = df_median.sort_values(["para", "job"]).reset_index(drop=True)

    # DF を dict に変換
    result_dict = _df_to_dict(df_median)
    del df_median
    return result_dict


def makeTrain(dataset: int, side: str, label: int, test=False):
    logger = getLogger("make_train").getChild("Make_json")
    logger.debug("PARAMETER", extra={"addinfo": f"dataset={dataset}, side={side}, label={label}"})

    # テスト用
    if test:
        file_name = "test"
    else:
        file_name = "snap_labels"

    # ファイルの生成
    if not os.path.exists(ML_DATA_DIR + f"/LIC_labels/{file_name}.json"):
        logger.debug("CHECK", extra={"addinfo": "ファイル作成"})
        _create_json(file_name)

    # ラベルによって処理が異なる
    if label == 0:  # 反応なし
        result_dict = _set_n()
    elif 0 < label <= 2:  # x点、o点用
        result_dict = _set_xo(dataset, side, label)
    else:  # その他
        logger.debug("ERROR", extra={"addinfo": "label の値が間違っています"})
        raise ValueError

    # 保存
    folder = ML_DATA_DIR + f"/LIC_labels/{file_name}.json"
    logger.debug("SAVE", extra={"addinfo": f"{folder} に保存"})
    with open(folder, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data == {}:
        _set_default()

    with open(folder, "w", encoding="utf-8") as f:
        data[str(dataset)][side][LABELS[label]] = result_dict
        json.dump(data, f)

    logger.debug("SAVE", extra={"addinfo": "完了"})


if __name__ == "__main__":
    from config.SetLogger import logger_conf

    logger = logger_conf("make_train")

    # 各種パラメータ
    logger.debug("START", extra={"addinfo": "処理開始"})
    # test = True
    test = False

    for dataset in DATASETS:
        for side in SIDES:
            for label in LABELS.keys():
                makeTrain(dataset, side, label, test=test)

    logger.debug("END", extra={"addinfo": "処理終了"})
