# -*- coding utf-8, LF -*-

import json
import os
import sys
from math import floor

import pandas as pd

sys.path.append(os.getcwd())

from config.params import ML_DATA_DIR, datasets, labels


def create_json(file_name):
    json_dict = dict()
    for dataset in datasets:
        json_dict[dataset] = dict()

        for side in ["left", "right"]:
            json_dict[dataset][side] = dict()

            for label in labels:
                json_dict[dataset][side][label] = list()

    with open(ML_DATA_DIR + f"/LIC_labels/{file_name}.json", "w", encoding="utf-8") as f:
        json.dump(json_dict, f)


def _df_to_dict(df: pd.DataFrame):
    """DataFrame を json に変換する関数

    return
        save_list[0] == {
            "job": 7,
            "params": {
                "x_range": [13, 38],
                "y_center": {"1": 287, "2": 289, "3": 294, "4": 298, "5": 308, "6": 315, "7": 319, "8": 316}, "y_range": {"1": [262, 312], "2": [264, 314], "3": [244, 344], "4": [248, 348], "5": [258, 358], "6": [265, 365], "7": [269, 369], "8": [266, 366]},
                "shape": {"1": [25, 50], "2": [25, 50], "3": [25, 100], "4": [25, 100], "5": [25, 100], "6": [25, 100], "7": [25, 100], "8": [25, 100]},
                "x_center": 26
            }
        }
    """
    save_list = list()  # 保存用のリスト

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

        save_list[i] = save_dict
    return save_list


def makeTrain(dataset, side, label, test=False):
    # 各種パラメータ
    labels = ["n", "x", "o"]
    params = [1, 2, 3, 4, 21, 22, 23, 24]

    if test:
        file_name = "test"
    else:
        file_name = "snap_labels"

    # ファイルの生成
    if not os.path.exists(ML_DATA_DIR + f"/LIC_labels/{file_name}.json"):
        create_json(file_name)

    # データのロード
    df_snap = pd.read_csv(ML_DATA_DIR + f"/LIC_labels/label_snap{dataset}_org.csv")

    # 加工
    df_snap = df_snap[(df_snap["dataset"] == dataset) & (df_snap["side"] == side) & (df_snap["label"] == label)]
    df_snap["xlow"] = df_snap["xlow"].map(lambda x: floor(x))  # 小数の切り上げ
    df_snap = df_snap[df_snap["job"] >= 7]  # 6 以下は使わない

    # サイズの計算
    df_snap["width"] = df_snap["xup"] - df_snap["xlow"]  # 幅
    df_snap["height"] = df_snap["yup"] - df_snap["ylow"]  # 高さ

    # パラメータ毎にデータを整形
    columns = ["dataset", "para", "job", "centerx", "centery", "xlow", "xup", "ylow", "yup", "width", "height", "label"]
    df_snap_list = [df_snap[df_snap["para"] == i][columns].reset_index(drop=True) for i in params]

    # job, 位置固定、param 変動
    columns = ["para", "job", "centerx", "centery", "xlow", "xup", "ylow", "yup", "width", "height"]
    df_index = []  # 各画像で同位置のもの(ex. 左から一つ目のO点) を集計
    df_all = pd.concat(df_snap_list)  # 全結合
    for i in range(len(df_snap_list[0])):
        df_a = df_all[df_all.index == i][columns]
        df_a["xlow_diff"] = df_a["xlow"] - df_a["xlow"].mean()
        df_a["xup_diff"] = df_a["xup"] - df_a["xup"].mean()
        df_a["ylow_diff"] = df_a["ylow"] - df_a["ylow"].mean()
        df_a["yup_diff"] = df_a["yup"] - df_a["yup"].mean()
        df_a["xcenter_diff"] = df_a["centerx"] - df_a["centerx"].mean()
        df_a["ycenter_diff_diff"] = df_a["centery"] - df_a["centery"].mean()
        df_index.append(df_a)

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

    # 保存
    folder = ML_DATA_DIR + f"/LIC_labels/{file_name}.json"
    with open(folder, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(folder, "w", encoding="utf-8") as f:
        data[str(dataset)][side][labels[label]] = result_list
        json.dump(data, f)


if __name__ == "__main__":
    # 各種パラメータ
    dataset = 77
    side = "left"
    label = 2

    makeTrain(dataset, side, label)
