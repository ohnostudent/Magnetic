#　教師データの作成用。ビューワと併用する。

from types import MethodDescriptorType
from flask import Flask, jsonify,request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, supports_credentials=True)
predoc = " "
@app.route("/postdata", methods=["POST"])
def writedata():
    req = request.json[0]
    response = {"status":"success", "message":""}
    try:
        with open(req["filepath"], "a") as f:
            # f.write(str(request.json[0]))
            #{'snappath': './snap49/lic_snap49.52.03.bmp',
            # 'filepath': '../txt/test.csv
            #  'locnumx': 0, 'locnumy': 0,
            #  'locnumx2': 0, 'locnumy2': 0,
            #  'rangenumx': 0, 'rangenumy': 0}
            doc = str(request.json[0])
            # snappath,dataset,para,job,center[x,y],xrange[a,z], yrange[a,z]
            dataset, para, job = req['snappath'][-12:-10], req['snappath'][-9:-7], req['snappath'][-6:-4] 
            centerx = req["locnumx"]
            centery = req["locnumy"]
            xlow = req["locnumx2"]
            xup = int(req["locnumx2"]) + int(req["rangenumx"])
            ylow = req["locnumy2"]
            yup = int(req["locnumy2"]) + int(req["rangenumy"])
            doc = f"{req['snappath']},{dataset},{para},{job},{centerx},{centery},{xlow},{xup},{ylow},{yup}\n"
            global predoc
            if int(req["rangenumx"]) == 0 or int(req["rangenumy"]) == 0:
                # return "error:range is 0"
                response["message"] = "error:range is 0"
                response["status"] = "error"
            elif doc != predoc:
                predoc = doc
                f.write(doc)
                # return "succes"
                response["message"] = "success"
                response["status"] = "success"

            else:
                # return "error:double reqest"
                response["message"] = "error:double reqest"
                response["status"] = "success"
        # return "succes"

    except FileNotFoundError:
        # return f"File not Found{req}"
        response["message"] = f"File not Found{req}"
        response["status"] = "error"
    except:
        # return f"failed{req['snappath']}"
        response["message"] = f"failed{req['snappath']}"
        response["status"] = "error"
    return jsonify(response)

app.run(host="127.0.0.1", port=8888, debug=True)