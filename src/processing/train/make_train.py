# -*- coding utf-8, LF -*-

import json
import os
import sys
from math import floor

import pandas as pd

sys.path.append(os.getcwd())

from config.params import ML_DATA_DIR, datasets, labels, sides


def _create_json(file_name):
    """基盤の作成を行う関数

    Args:
        file_name (str): 保存するファイルの名前
    """
    json_dict = _set_default()
    # 保存
    with open(ML_DATA_DIR + f"/LIC_labels/{file_name}.json", "w", encoding="utf-8") as f:
        json.dump(json_dict, f)

def _set_default():
    # 基盤の作成
    json_dict = dict()
    for dataset in datasets:
        json_dict[dataset] = dict()

        for side in sides:
            json_dict[dataset][side] = dict()

            for label in labels:
                json_dict[dataset][side][label] = dict()
    return json_dict



def _df_to_dict(df: pd.DataFrame):
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
    result_list = dict()  # 保存用のリスト

    for i in range(7, 15):
        params_dict = df[df["para"] == i].reset_index(drop=True).to_dict()

        save_dict = dict()
        save_dict["x_range"] = list()
        save_dict["y_center"] = dict()
        save_dict["y_range"] = dict()
        save_dict["shape"] = dict()

        for j in range(8):  # 8点
            save_dict["x_center"] = params_dict["centerx"][j]
            save_dict["x_range"] = [params_dict["xlow"][j], params_dict["xup"][j]]
            save_dict["y_center"][j] = params_dict["centery"][j]
            save_dict["y_range"][j] = [params_dict["ylow"][j], params_dict["yup"][j]]
            save_dict["shape"][j] = [params_dict["width"][j], params_dict["height"][j]]

        result_list[i] = save_dict

    return result_list


def set_n():
    # 切り取る長方形の重心座標のリスト
    # 4 * 2 = 8 点取る
    x_locs = [30, 90, 150, 210]
    y_locs = [150, 450]
    shapes = [25, 100]

    save_dict = dict() # 保存用の辞書
    save_dict["x_center"] = dict()
    save_dict["x_range"] = dict()
    save_dict["y_center"] = dict()
    save_dict["y_range"] = dict()
    save_dict["shape"] = dict()

    for i in range(8):
        x_idx = i % 4
        y_idx = i % 2

        save_dict["x_center"][i] = x_locs[x_idx]
        save_dict["x_range"][i] = [x_locs[x_idx] - 13, x_locs[x_idx] + 12]
        save_dict["y_center"][i] = y_locs[y_idx]
        save_dict["y_range"][i] = [y_locs[y_idx] - shapes[1] // 2, y_locs[y_idx] + shapes[1] // 2]
        save_dict["shape"][i] = shapes

    return save_dict


def set_xo(dataset: int, side: str, label: int):
    # データのロード
    df_snap = pd.read_csv(ML_DATA_DIR + f"/LIC_labels/label_snap{dataset}_org.csv")
    params = [1, 2, 3, 4, 21, 22, 23, 24]

    # 加工
    df_snap = df_snap[(df_snap["dataset"] == dataset) & (df_snap["side"] == side) & (df_snap["label"] == label)]
    df_snap["xlow"] = df_snap["xlow"].map(lambda x: floor(x))  # 小数の切り上げ
    df_snap = df_snap[df_snap["job"] >= 7]  # 6 以下は使わない

    # サイズの計算
    df_snap["width"] = df_snap["xup"] - df_snap["xlow"]  # 幅
    df_snap["height"] = df_snap["yup"] - df_snap["ylow"]  # 高さ

    # パラメータ毎にデータを整形
    columns = ["dataset", "para", "job", "centerx", "centery", "xlow", "xup", "ylow", "yup", "width", "height", "label"]
    df_snap_list = [df_snap[df_snap["para"] == i][columns].sort_values("centerx").reset_index(drop=True) for i in params]

    # job, 位置固定、param 変動
    columns = ["para", "job", "centerx", "centery", "xlow", "xup", "ylow", "yup", "width", "height"]
    df_index = []  # 各画像で同位置のもの(ex. 左から一つ目のO点) を集計
    df_all = pd.concat(df_snap_list)  # 全結合
    for i in range(len(df_snap_list[0])):
        df_snap_i = df_all[df_all.index == i][columns]
        df_snap_i["xlow_diff"] = df_snap_i["xlow"] - df_snap_i["xlow"].mean()
        df_snap_i["xup_diff"] = df_snap_i["xup"] - df_snap_i["xup"].mean()
        df_snap_i["ylow_diff"] = df_snap_i["ylow"] - df_snap_i["ylow"].mean()
        df_snap_i["yup_diff"] = df_snap_i["yup"] - df_snap_i["yup"].mean()
        df_snap_i["xcenter_diff"] = df_snap_i["centerx"] - df_snap_i["centerx"].mean()
        df_snap_i["ycenter_diff_diff"] = df_snap_i["centery"] - df_snap_i["centery"].mean()
        df_index.append(df_snap_i)

    del df_snap_list

    # 上の要約
    df_describe_all = list()
    for i in range(len(df_index)):
        df_describe: pd.DataFrame = df_index[i]
        df_describe_all.append(df_describe.describe().round(3))

    # 中央値(50%)を基に基準値を作成
    columns = ["para", "job", "centerx", "centery", "xlow", "xup", "ylow", "yup", "width", "height"]
    df_median = pd.DataFrame(columns=columns)
    for df_describe_i in df_describe_all:
        df_median = pd.concat([df_median, df_describe_i.loc[["50%"], columns]])
    del df_describe_all

    df_median = df_median.astype(int).reset_index(drop=True).reset_index()
    df_median["para"] = df_median["index"] % 8 + 7
    df_median["width"] = df_median["xup"] - df_median["xlow"]
    df_median["height"] = df_median["yup"] - df_median["ylow"]
    df_median = df_median.set_index("index")
    df_median = df_median.sort_values(["para", "job"]).reset_index(drop=True)

    # DF を dict に変換
    result_list = _df_to_dict(df_median)
    del df_median
    return result_list


def makeTrain(dataset: int, side: str, label: int, test=False):
    # テスト用
    if test:
        file_name = "test"
    else:
        file_name = "snap_labels"

    # ファイルの生成
    if not os.path.exists(ML_DATA_DIR + f"/LIC_labels/{file_name}.json"):
        _create_json(file_name)

    # ラベルによって処理が異なる
    if label == 0:  # 反応なし
        result_list = set_n()
    elif 0 < label <= 2:  # x点、o点用
        result_list = set_xo(dataset, side, label)
    else: # その他
        raise ValueError

    # 保存
    folder = ML_DATA_DIR + f"/LIC_labels/{file_name}.json"
    with open(folder, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data == {}:
        _set_default()

    with open(folder, "w", encoding="utf-8") as f:
        data[str(dataset)][side][labels[label]] = result_list
        json.dump(data, f)


if __name__ == "__main__":
    # from config.params import sides

    # for dataset in datasets:
    #     for side in sides:
    #         for label in range(3):
    #             makeTrain(dataset, side, label)

    if len(sys.argv) == 1:
        sys.exit()
    else:
        dataset = sys.argv[1]

    from config.params import set_dataset
    from config.SetLogger import logger_conf

    # logger = logger_conf()
    dataset = set_dataset(dataset)

    # 各種パラメータ
    side = "right" # "left", "right"
    label = 2
    test = True
    makeTrain(dataset, side, label, test=test)
