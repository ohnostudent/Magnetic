# 　教師データの作成用。ビューワと併用する。

import os

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, supports_credentials=True)
predoc = " "


@app.route("/postdata", methods=["POST"])
def writedata() -> Response:
    req = request.json[0]
    response = {"status": "success", "message": ""}
    try:
        with open(req["filepath"], "a") as f:
            # f.write(str(request.json[0]))
            # {'snappath': './images/LIC/snap77/left/lic_snap77.left.magfield.01.07.bmp',
            # 'filepath': '../txt/test.csv
            #  'locnumx': 0, 'locnumy': 0,
            #  'locnumx2': 0, 'locnumy2': 0,
            #  'rangenumx': 0, 'rangenumy': 0}

            doc = str(request.json[0])
            # snappath, dataset, para, job, side, center[x, y], xrange[a, z],  yrange[a, z]
            file_name = os.path.basename(req["snappath"]).split(".")
            dataset, para, job, side = file_name[0].split("_")[1].replace("snap", ""), file_name[3], file_name[4], file_name[1]
            centerx = req["locnumx"]
            centery = req["locnumy"]
            xlow = req["locnumx2"]
            xup = int(req["locnumx2"]) + int(req["rangenumx"])
            ylow = req["locnumy2"]
            yup = int(req["locnumy2"]) + int(req["rangenumy"])
            label = req["label"]
            doc = f"{req['snappath']},{dataset},{para},{job},{side},{centerx},{centery},{xlow},{xup},{ylow},{yup},{label}\n"
            global predoc

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
        response["message"] = f"failed{req['snappath']}"
        response["status"] = str(e)

    return jsonify(response)


app.run(host="127.0.0.1", port=8888, debug=True)
