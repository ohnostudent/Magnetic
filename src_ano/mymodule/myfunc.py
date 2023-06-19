import py_compile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import subprocess
from dotenv import load_dotenv
load_dotenv()


imgout = "../imgout/"
root_dir = "../"


def AVSlat2bilat(dataname, xy):
    dataname = str(dataname)
    xy = np.array(xy)
    bixy = np.array([513,1025])
    if dataname == "77":
        start = np.array([40,120])
        end = np.array([4759,879])
    else:
        print("このデータにはまだ対応していない。AVSで有効な座標を調べよ")
        return None
    leng = end - start + 1
    res = [0,0]
    for i in [0,1]:
        res[i] = (xy[i] - start[i]) * ( bixy[i] / leng[i])
    return res
def LIClat2bilat(xy, half=True):
    xy = np.array(xy)
    if half:
        bixy = np.array([257,1025])
    else:
        bixy = np.array([513,1025])

    start = np.array([0,0])
    end = np.array([1799,570])
    leng = end - start + 1
    res = [0,0]
    for i in [0,1]:
        res[i] = (xy[i] - start[i]) * ( bixy[i] / leng[i])
    return res
        


def gen_snap_path(target, para, job, dataset=49):
    if dataset == 49:
        snap_path = root_dir + "snap/snap49/"
    elif dataset == 77:
        snap_path = root_dir + "snap/snap77/"
    elif dataset ==497:
        snap_path = root_dir + "snap/snap497/"
    res = f"{snap_path}{target}/{'{0:02d}'.format(job)}/{target}.{'{0:02d}'.format(para)}.{'{0:02d}'.format(job)}"
    if os.path.exists(res):
        return res
    else:
        return None
#データのロード
def load(filename, z=3):
    """
    little endianのデータの読み込み。z=1でz方向が一層だけのデータを
    z=3でz方向が3のデータを1層だけ読み込む
    """
    if filename[-3:] == "npy":
        return np.load(filename)
    f = open(filename,mode='rb')
    #:525825でz方向の1個目だけ(xy平面一つ)とる。reshapeでx,yの整形
    if z == 1:
        # print(np.fromfile(f, dtype='f',sep='').shape)
        data = np.fromfile(f, dtype='f',sep='').reshape(1025,513)
    elif z == 3:
        data = np.fromfile(f, dtype='f',sep='')[:525825].reshape(1025,513)
    f.close()
    return data

def load_bigendian(filename):
    """
    fort.111.0等のビッグエンディアンのデータをリトルエンディアンに変換し、
    必要な部分を抜き取る。
    """
    f = open(filename,mode='rb')
    #>fはbig endianのfloat型。:525825でz方向の1個目だけ(xy平面一つ)とる。reshapeでx,yの整形
    data = np.fromfile(f, dtype='>f',sep='')[:525825].reshape(1025,513)
    f.close()
    return data
#ヒートマップの生成と保存
def show(data: np.array, imgname=False, bar_range=None):
    plt.clf()
    if bar_range == None:
        sns.heatmap(data)
    else:
        sns.heatmap(data, vmin=bar_range[0], vmax=bar_range[1])

    if imgname:
        plt.savefig(f"{imgname}")


#畳み込み
def convolute(data: np.array, carnel: np.array, padding=0, stride=1):
    """
    畳み込み演算。
    """
    
    if padding:
        print("0パディングの処理未実装  padding=0で実行します")
        padding = 0
    
    c_width = carnel.shape[1]
    c_height = carnel.shape[0]

    result_width = int((data.shape[1] + 2*padding - carnel.shape[1]) / stride + 1)
    result_height = int((data.shape[0] + 2*padding - carnel.shape[0]) / stride + 1)
    # print("result   Y,X:",result_height,result_width)
    # print("original Y,X:",data.shape)
    def loop_calc():
        def calc(array):

            result = sum(array*carnel)
            result = sum(result.flat)
            return result
        orgX = 0
        orgY = 0
        convoluted = np.zeros((result_height, result_width))

        for resultY in range(result_height):
            for resultX in range(result_width):
                array = data[orgY : orgY + c_height, orgX : orgX + c_width]
                # a = convoluted[resultY]
                convoluted[resultY][resultX] = calc(array)
                orgX += stride

            orgX = 0
            orgY += stride
        return convoluted

    return loop_calc()

def ave_carnel(size:int):
    """
    畳み込みにおける平滑化のカーネル作成
    """
    ones = np.ones((size,size))
    res = ones / (size**2)
    return res

#離散データの微分
def diff(x, h):
    """
    精度低めの普通の微分。誤差:h**2
    """
    res = x[2:] - x[:-2]
    # print(x[1:])
    # print(x[:-1])
    return res/(2*h)
def diff4(x, h):
    """
    精度高めの微分・。誤差:h**4
    1回微分{-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)}/12h
    xは時系列データ,hはデータ間の時間(second)
    ベクトル長が4短くなる
    """
    res = -x[4:] + 8*x[3:-1] - 8*x[1:-3] + x[:-4]
    return res/(12*h)


def diff4_x(data: np.ndarray, h:float):
    """
    diff4を使った行列の横方向偏微分
    """
    res = np.ndarray([])
    for vecx in data:
        if res.shape == tuple():
            res = diff4(vecx, h)
        else:
            res = np.append(res,diff4(vecx, h))
    return res.reshape(data.shape[0],data.shape[1]-4)
def diff4_y(data: np.ndarray, h:float):
    """
    diff4を使った行列の縦方向偏微分
    """
    data = data.T
    res = np.ndarray([])
    for vecy in data:
        if res.shape == tuple():
            res = diff4(vecy, h)
        else:
            res = np.append(res,diff4(vecy, h))
    return res.reshape(data.shape[0],data.shape[1]-4).T
def rot2d(vX:np.ndarray,vY:np.ndarray):
    """
    vX,vYよりローテーションを出す。出力はz方向のベクトル
    y方向については、vZ(zx平面)、vX(zx平面)の順で入力、x方向はvY(yz平面)、vZ(yz平面)の順で入力すると得られる。
    x､y､zそれぞれの出力であるスカラーの行列に対して、
    それぞれの方向の単位ベクトルを掛けて足せば3次元のローテーションが求まる
    """
    return diff4_x(vY,1)[2:-2,] - diff4_y(vX,1)[:,2:-2]

"""
rot2dの結果に対して単位ベクトルを掛けるやり方。
a = np.array([[0,1,2,3],[0,4,5,6],[0,7,8,9]])
e = np.array([1,0,0])
mylist = [[j *e for j in i ] for i in a]
"""
from scipy import interpolate#x方向少なすぎるのか上手く動かない、linearは縦線入ってまう
def im_interpolate(data, gridx, gridy, kind = "cubic"):
    XLEN = 513
    YLEN = 1025
    dx = 2*np.pi/XLEN
    dy = 2/YLEN
    datay, datax = data.shape
    x = np.arange(0, datax*dx, dx)
    y = np.arange(0, datay*dy, dy)
    X,Y = np.meshgrid(x, y)
    f = interpolate.interp2d(X, Y, data, kind=kind)
    xnew = np.linspace(0, datax*dx, gridx)
    ynew = np.linspace(0, datay*dy, gridy)
    znew = f(xnew, ynew)
    # xnew_,ynew_=np.meshgrid(xnew, ynew)
    return znew

from struct import pack
import random
def ohno_stream(xfile:str, yfile:str, outname:str, x = False, y = False):
    command = [f"{root_dir}src/mymodule/StreamLines/FieldLines.exe", xfile, yfile, outname]
    if xfile[-3:] == "npy":
        xdata = np.load(xfile)
        ydata = np.load(yfile)
        command += list(map(str, list(reversed(xdata.shape))))
        #npy1次元のファイルを作って無理やり読み込ます。そして消す。名前はランダムにして
        xtempfile = f"xtemp_ohnostrm_reading{random.randint(10000, 99999)}"
        f = open(xtempfile, "ab")
        for val in list(xdata.flat)*3:#*3は元のデータがz軸方向に3重なっているのを表現
            b = pack('f', val)
            f.write(b)
        f.close()
        ytempfile = f"ytemp_ohnostrm_reading{random.randint(10000, 99999)}"
        f = open(ytempfile, "ab")
        for val in list(ydata.flat)*3:#*3は元のデータがz軸方向に3重なっているのを表現
            b = pack('f', val)
            f.write(b)
        f.close()
        command[1], command[2] = xtempfile, ytempfile
        res = subprocess.run(command)
        os.remove(xtempfile)
        os.remove(ytempfile)
    elif (x and y) == True:
        command += list(map(str, [x,y]))
        res = subprocess.run(command)
    else:
        res = subprocess.run(command)
    print(command)
    return res

def ohno_lic(xfile:str, yfile:str, outname:str, x = False, y = False):
    command = [f"{root_dir}src/mymodule/LIC/LIC.exe", xfile, yfile, outname]
    if xfile[-3:] == "npy":
        xdata = np.load(xfile)
        ydata = np.load(yfile)
        command += list(map(str, list(reversed(xdata.shape))))
        #npy1次元のファイルを作って無理やり読み込ます。そして消す。名前はランダムにして
        xtempfile = f"xtemp_ohnostrm_reading{random.randint(10000, 99999)}"
        f = open(xtempfile, "ab")
        for val in list(xdata.flat)*3:#*3は元のデータがz軸方向に3重なっているのを表現
            b = pack('f', val)
            f.write(b)
        f.close()
        ytempfile = f"ytemp_ohnostrm_reading{random.randint(10000, 99999)}"
        f = open(ytempfile, "ab")
        for val in list(ydata.flat)*3:#*3は元のデータがz軸方向に3重なっているのを表現
            b = pack('f', val)
            f.write(b)
        f.close()
        command[1], command[2] = xtempfile, ytempfile
        res = subprocess.run(command)
        os.remove(xtempfile)
        os.remove(ytempfile)
    elif (x and y) == True:
        command += list(map(str, [x,y]))
        res = subprocess.run(command)
    else:
        res = subprocess.run(command)
    print(command)
    return res

def sort_paths(pathlist, paraloc=[-9,-8], jobloc=[-6,-5]):
    """
    pathlistはglob等で取得したlist。
    locはそれぞれのpath上での位置。
    pathlistをpara,jobでソートして返す。
    """
    pjp = [{"path": path,
     "job": int("".join([path[i] for i in jobloc])),
     "para": int("".join([path[i] for i in paraloc]))} for path in pathlist]
    pjp2 = sorted(pjp, key = lambda x: (x["para"]))
    pjp3 = sorted(pjp2, key = lambda x: (x["job"]))
    return [x["path"] for x in pjp3]
from PIL import Image
def li2image(lic_result, xzoom=3, yzoom=1):
    shape = lic_result.shape
    im = np.array(lic_result*255, dtype="uint8")
    im = Image.fromarray(im, mode="L")
    
    im = im.resize([shape[1]*xzoom,shape[0]*yzoom], resample=Image.LANCZOS)#######
    # im.show()
    # im.save(out)

def resize(array, yx):#返値がyxの形と同じだったり逆だったりする。
    if list(array.shape) == list(yx):
        return array
    im = Image.fromarray(array, mode="L")
    im = im.resize([yx[0], yx[1]], resample=Image.LANCZOS)#######
    return np.array(im)

    
