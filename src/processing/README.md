# データの加工

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

## 2. データの加工
1.  元データを各種パラメータに分割する
    - /src/Processing 配下にある`separator.py` を実行する
    - 元データをそれぞれのデータに分割するプログラム
    - `mkdirs.bat` にてディレクトリの生成を一括で行っている
    - 各データファイルの生成処理毎に、生成したファイルを移動している
    - `.py`　と `.ipynb` では作業ディレクトリの位置が違うので要注意
    - 出力先：`/snaps/snap{i}{j}/*/*`

    ```cmd
    ./Magnetic> python ./src/Processing/separator.py

    ```
    ```python
    import os
    import sys
    import shutil
    import subprocess
    from glob import glob

    sys.path.append(os.getcwd())

    from config.params import BIN_PATH, ROOT_DIR, SNAP_PATH, SRC_PATH, datasets
    from Processing.separatorLoop import move_file, rename_file, set_ij

    # パラメータの定義
    items1 = ["density", "enstrophy", "pressure", "magfieldx", "magfieldy", "magfieldz", "velocityx", "velocityy", "velocityz"]
    items2 = ["magfield1", "magfield2", "magfield3", "velocity1", "velocity2", "velocity3"]
    xyz = {1: "x", 2: "y", 3: "z"}

    for dataset in datasets:
        ij = set_ij(dataset)
        if ij:
            i, j = ij
        else:
            print("Value Error", "入力したデータセットは使用できません")
            sys.exit()

        # bat ファイルの実行
        # 基本的に加工したデータの保存先のフォルダの作成
        print("MAKE", "ディレクトリの作成")
        subprocess.run([BIN_PATH + "/Snaps.bat", str(dataset)])

        # ログの保存先
        files = glob(ROOT_DIR + f"/data/ICh.dataset=50.ares=1.0d-{i}.adiffArt=1.0d-{j}.h00.g00.BCv1=0.0/Snapshots/*")
        for file in files:
            # 元データの分割処理の実行
            subprocess.run([SRC_PATH + "/Processing/cln/separator.exe", f"{file}"])
            _, _, _, param, job = map(lambda x: int(x) if x.isnumeric() else x, os.path.basename(file).split("."))
            print("OPEN", f"File snap{i}{j}.{param:02d}.{job:02d}")

            # 出力されたファイル名の変更
            for item2 in items2:
                if os.path.exists(item2):
                    rename_file(xyz, param, job, item2)

                else:  # 見つからない場合
                    print("NotFound", f"ファイル {item2}.{param:02d}.{job:02d}")

            # 出力されたファイルの移動
            for item1 in items1:
                if os.path.exists(item1):
                    move_file(dataset, item1)

                else:  # 見つからない場合
                    print("NotFound", f"ファイル {item2}.{param:02d}.{job:02d}")

            print("CLOSE", f"File {item2}.{param:02d}.{job:02d}")

        # coordn を最後に移動させる
        for i in range(1, 4):
            shutil.move("coord" + xyz[i], SNAP_PATH + f"/snap{dataset}")

        print("END", "処理終了")

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
    import os
    import sys
    from glob import glob

    sys.path.append(os.getcwd() + "/src")

    from config.params import SNAP_PATH, datasets, variable_parameters, set_dataset
    from Processing.snap2npy import snap2npy
    dataset = set_dataset(input())

    sp = SnapData()
    print("START", f"Snap{dataset} 開始")

    for param in variable_parameters:
        print("START", f"{param} 開始")

        for path in glob(SNAP_PATH + f"/snap{dataset}/{param}/*/*"):
            # print(path)
            snap2npy(sp, path, dataset)

        print("END", f"{param} 終了")
    print("END", f"Snap{dataset} 終了")

    ```

<br>
<br>

## 4. 機械学習
### 4.1. 教師データの作成

#### 4.1.2 データの切り取り
- `/src/Processing/viewer/fusion_npy.py` を実行
- 教師データの作成

```python
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/src")

from config.params import set_dataset
from Processing.fusion_npy import CrateTrain


dataset = set_dataset(input())

md = CrateTrain()
for val in ["magfieldx", "magfieldy", "velocityx", "velocityy", "density"]:
    md.cut_and_save(dataset, val)
```

#### 4.1.3 データの合成
- 画像

```python
import os
import sys
from glob import glob

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/src")

from config.params import ML_DATA_DIR, labels, set_dataset
from Processing.fusion_npy import CrateTrain


dataset = set_dataset(input())

md = CrateTrain()
props_params = [
    (["magfieldx", "magfieldy"], "mag_tupledxy", md.kernellistxy),
    (["velocityx", "velocityy", "density"], "energy", md.kernelEnergy),
]
OUT_DIR = ML_DATA_DIR + f"/snap{dataset}"

# /images/0131_not/density/density_49.50.8_9.528
for val_params, out_basename, kernel in props_params:
    for label in labels:
        npys = OUT_DIR + f"/point_{label}"

        for img_path in glob(npys + "/" + val_params[0] + "/*.npy"):
            im_list = md.loadBinaryData(img_path, val_params)  # 混合データのロード
            resim = kernel(*im_list)  # データの作成

            # 保存先のパスの作成
            # /MLdata/snap{dataset}/{out_basename}/{out_basename}_{dataset}.{param}.{job}_{centerx}.{centery}.npy
            # /MLdata/snap77/energy/energy_77.01.03_131.543.npy
            out_path = npys + "/" + out_basename + "/" + os.path.basename(img_path).replace(val_params[0], out_basename)
            md.saveFusionData(resim, out_path)  # データの保存

```