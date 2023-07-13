# 磁気リコネクション
<right>written by Kuniya Ota</right>

23卒の先輩方が行った研究のデータをもとに、リファクタリングを行ったもの

※ /Magnetic を作業ディレクトリとしてください
```
> pwd  
./Magnetic  

```
※ 場合によっては以下のコードを追記してください
```python
import os
import sys
sys.path.append(os.getcwd() + "/src")

```
<br>

## 1. 元データ
1. コマンドラインにて以下のコードを実行する
    ```
    pip install -r requirements.txt
    ```

2. `/etc/mkdir.bat` を実行する  

3. 元データを `./data` 配下に保存する  
    ```
    /data/ICh.target=50.ares=1.0d-{i}.adiffArt=1.0d-{j}.h00.g00.BCv1=0.0
    ```
    となっているフォルダが3つ存在するため、`i`, `j` からフォルダ名を`snap+{i+j}`とした

4. 元データの中身
    - 縦1025 * 横513 のデータ
    - 各種パラメータ
        1. density : 密度
        1. enstrophy : エンストロフィー
        1. magfieldx : 磁場x方向
        1. magfieldy : 磁場y方向
        1. magfieldz : 磁場z方向
        1. pressure : 圧力
        1. velocityx : 速度x方向
        1. velocityy : 速度y方向
        1. velocityz : 速度z方向
<br>
<br>

## 2. データの加工
1.  元データを各種パラメータに分割する
    - /src/Processing 配下にある`separater.py` を実行する
    - 元データをそれぞれのデータに分割するプログラム
    - `mkdirs.bat` にてディレクトリの生成を一括で行っている
    - 各データファイルの生成処理毎に、生成したファイルを移動している
    - `.py`　と `.ipynb` では作業ディレクトリの位置が違うので要注意
    - 出力先：`/snaps/snap{i}{j}/*/*`

    ```cmd
    ./Magnetic> python ./src/Processing/separater.py

    ```
<br>
<br>

2. binary を .npy に変換
    - `/src/Processing/snap2npy.py` を実行する
    - numpy に変換し、教師データの元にする
    - それなりに時間がかかる
    - 縦1025 * 横513 のデータを、縦625 (200~825)(上下200を切り取る) * 横257 (左右 保存) に加工している
<br>
<br>
    ```cmd
    ./Magnetic> python ./src/Processing/snap2npy.py

    ```
    ```python
    from params import SNAP_PATH, datasets, variable_parameters
    from Processing.snap2npy import snap2npy

    sp = SnapData()

    for dataset in datasets:
        for val_param in variable_parameters:
            for path in glob(SNAP_PATH + f"/snap{dataset}/{val_param}/*/*"):
                # print(path)
                snap2npy(sp, path, dataset)
    ```

<br>
<br>

## 3. 流線の可視化
### 1. Heatmap
- 使用メソッド
    - plt.streamplot
    - plt.contour
    - sns.heatmap
    - cv2.cvtColor

    出力先：
    - `/imgout/visualization/heatmap`
    - `/imgout/visualization/edges`
    - `/imgout/visualization/Energy_magfield`
    - `/imgout/visualization/Energy_vectory`
<br>
<br>

    ```python：/src/Visualization
    from glob import glob
    from src.params import SNAP_PATH, datasets
    from src.SetLogger import logger_conf

    # ログ取得の開始
    logger = logger_conf()

    for dataset in datasets:
        logger.debug("START", extra={"addinfon": f"snap{dataset}"})
        target_path = SNAP_PATH + f"/snap{dataset}"

        # インスタンスの生成
        viz = VisualizeMethod(dataset)

        files = {} # glob した path の保存

        # エネルギーの速さと密度の可視化
        files["density"] = glob(target_path + f"/density/*/*")
        files["velocityx"] = glob(target_path + f"/velocityx/*/*")
        files["velocityy"] = glob(target_path + f"/velocityy/*/*")
        for dens_path, vx_path, vy_path in zip(files["density"], files["velocityx"], files["velocityy"]):
            viz.drawEnergy_for_velocity(dens_path, vx_path, vy_path)

        # エネルギーの磁場の可視化
        files["magfieldx"] = glob(target_path + f"/magfieldx/*/*")
        files["magfieldy"] = glob(target_path + f"/magfieldy/*/*")
        for magx_path, magy_path in zip(files["magfieldx"], files["magfieldy"]):
            viz.drawEnergy_for_magfield(magx_path, magy_path)
        
        # Heatmap と edge の可視化
        files["enstrophy"] = glob(target_path + f"/enstrophy/*/*")
        for val_param in ["velocityx", "velocityy", "magfieldx", "magfieldy", "density", "enstrophy"]:
            for path in files[val_param]:
                viz.drawHeatmap(path)
                viz.drawEdge(path)

        logger.debug("END", extra={"addinfon": f"snap{dataset}"})

    ```

<br>
<br>

### 2. AVS
処理ファイル：`/src/AVS`  
出力先
- `/imgout/AVS/*`

<br>
<br>

### 3. StreamLine
処理ファイル：`/src/StreamLines`  
出力先
- `/imgout/StreanLines/*`

<br>
<br>

### 4. LIC
- LIC法にて可視化する
- 縦625 * 横256 の加工済みnpyデータを使う
- snaps/half_left/snap77/magdfield に保存したデータをすべて処理するために、3.9GHz, 10並列で15時間程度かかる

    処理ファイル：`/src/LIC`  
    出力先
    - `/imgout/LIC/snap{i}/left/*`
    - `/imgout/LIC/snap{i}/right/*`

    ```python
    from params import SNAP_PATH, IMG_PATH, datasets
    from LIC.LIC import LIC

    logger.debug("START", extra={"addinfo": "処理開始"})

    lic = LIC()
    out_dir = IMGOUT + "/LIC"
    lic.makedir("/LIC")

    for dataset in datasets:
        indir = SNAP_PATH + f"/half/snap{dataset}"
        dir_basename = os.path.basename(indir) # snap77
        base_out_path = out_dir + "/" + os.path.basename(indir) # ./imgout/LIC/snap77
        lic.makedir(f"/LIC/snap{dataset}")

        binary_paths = glob(indir+"/magfieldx/*/*.npy")
        # ファイルが無い場合
        if binary_paths == []:
            raise "Error File not Found"
        
        for xfile in binary_paths[-1:]:
            yfile = xfile.replace("magfieldx", "magfieldy")
            out_path = base_out_path + f"/lic_{dir_basename}.{os.path.splitext(os.path.basename(xfile))[0]}.bmp"
            # print(out_path) # ./imgout/LIC/snap77/lic_snap77.magfieldx.01.14.bmp
            
            command = lic.set_command(xfile, yfile, out_path)
            lic.LIC(command)

    ```

<br>
<br>

## 4. 機械学習
### 1. 教師データの作成

#### 1-1 ビューワの作成
処理ファイル：`/src/Processing/viewer/createViewer.py`  
出力先
- `/MLdata/*`
    ```cmd
    ./Magnetic> python ./src/Processing/snap2npy.py

    ```
    ```python
    from Processing.viewer.createViewer import createViewer

    dataset = input("使用するデータセットを入力してください(77/497/4949) : ")
    if dataset not in datasets:
        sys.exit()

    createViewer(dataset)
    ```

#### 1-2 画像の分割
1. `/src/Processing/viewer/writer.py` を実行
2. 


#### 1-3


<br>
<br>

### 2. SVM

出力先
- `/MLres/*`

<br>
<br>

### 3. 非線形SVM

出力先
- `/MLres/*`

<br>
<br>

### 4. k-Means

出力先
- `/MLres/*`

<br>
<br>

### 5. XGBoost

出力先
- `/MLres/*`

<br>
<br>

### 6. CNN

出力先
- `/MLres/*`

<br>
<br>

## ディレクトリ構造
Magnetic/  

    ├ .git/  
    ├ data/  元データ  
    ├ imgout/  画像データの出力先  
    ├ MLdata/ 教師データ  
    ├ MLres/ 学習結果の保存先  
    ├ snaps/  パラメータ毎に分解したデータ  
    |   ├ density/  
    |   |     ├ 00/  
    |   |     ├ 01/  
    .   .     .  
    .   .     .  
    |   |     └14/  
    |   ├ enstrophy/  
    |   |     ├ 00/  
    |   |     ├ 01/  
    .   .     .  
    .   .     .  
    |   |     └14/  
    |   ├ magfieldx/  
    .   .  
    .   .  
    |   └velocityz  
    |  
    ├ src/  
    |   ├ AVS/ AVS可視化  
    |   ├ k_means/ k近傍法  
    |   ├ LIC/ LIC可視化  
    |   ├ Processing/ 元データの加工  
    |   ├ SteamLines/ Stream可視化  
    |   └ Visualization/ 元データの可視化  
    |   
    |   .  
    |   .   
    |  
    ├ .env  
    ├ main.ipynb  
    ├ README.md  
    └.gitignore  


```
make_data
-> labeling (bmp -> csv)
-> makesepnpy.ipynb (csv -> npy)
-> fusionnpy.ipynb (fusion npy)

ML
-> _clustering.ipynb (k-means)
-> MLs.py
-> doML.py
-> XGtune.ipynb
```
