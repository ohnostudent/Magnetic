# -*- coding utf-8, LF -*-

import os
import sys
from glob import glob

from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

sys.path.append(os.getcwd() + "/src")

from config.params import IMAGE_PATH

app = Flask(__name__, static_folder=IMAGE_PATH + "/LIC")
cors = CORS(app, supports_credentials=True)
predoc = " "


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
            map(lambda path: [path] + os.path.basename(path).split(".")[3:5], path_list),
        )
    )
    # params, job の順にソート
    pjp_sorted = sorted(pjp, key=lambda x: (x[2], x[1]))

    return list(map(lambda x: x[0].replace("\\", "/").replace("I:/GraduationResearch/Magnetic/images", ""), pjp_sorted))


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/viewer/<int:dataset>-<string:side>")
def viewer(dataset, side) -> str:
    path_list_str = _sort_paths(glob(IMAGE_PATH + f"/LIC/snap{dataset}/{side}/*.bmp"))
    return render_template("viewer_template.html", dataset=dataset, side=side, sortPaths=path_list_str)


@app.route("/postdata", methods=["POST"])
def writedata() -> Response:
    req = request.json[0]  # type: ignore
    response = {"status": "success", "message": ""}
    try:
        with open(req["filepath"], "a") as f:
            # f.write(str(request.json[0]))
            # {'snappath': './images/LIC/snap77/left/lic_snap77.left.magfield.01.07.bmp',
            # 'filepath': '../txt/test.csv
            #  'locnumx': 0, 'locnumy': 0,
            #  'locnumx2': 0, 'locnumy2': 0,
            #  'rangenumx': 0, 'rangenumy': 0}

            doc = str(request.json[0])  # type: ignore
            # snappath, dataset, para, job, side, center[x, y], xrange[a, z],  yrange[a, z]
            file_name = os.path.basename(req["snappath"]).split(".")
            dataset, para, job, side = file_name[0].split("_")[1].replace("snap", ""), file_name[3], file_name[4], file_name[1]
            centerx, centery = req["locnumx"], req["locnumy"]
            xlow, ylow = req["locnumx2"], req["locnumy2"]
            xup = int(req["locnumx2"]) + int(req["rangenumx"])
            yup = int(req["locnumy2"]) + int(req["rangenumy"])
            label = req["label"]
            doc = f"{req['snappath']},{dataset},{para},{job},{side},{centerx},{centery},{xlow},{xup},{ylow},{yup},{label}\n"
            global predoc  # noqa: PLW0603
            print(doc)

            if int(req["rangenumx"]) == 0 or int(req["rangenumy"]) == 0:
                # return "error:range is 0"
                response["message"] = "error:range is 0"
                response["status"] = "error"

            elif doc != predoc:
                predoc = doc
                f.write(doc)
                # return "success"
                response["message"] = "success"
                response["status"] = "success"

            else:
                # return "error:double request"
                response["message"] = "error:double request"
                response["status"] = "success"
        # return "success"

    except FileNotFoundError:
        # return f"File not Found{req}"
        response["message"] = f"File not Found{req}"
        response["status"] = "error"

    except Exception as e:
        # return f"failed{req['snappath']}"
        response["message"] = f"failed {req['snappath']}"
        response["status"] = str(e)

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8888, debug=True)
