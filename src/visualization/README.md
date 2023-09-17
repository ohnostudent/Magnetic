## 3. 流線の可視化

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

### 3.1. Heatmap
- 使用メソッド
    - plt.streamplot
    - plt.contour
    - sns.heatmap
    - cv2.cvtColor

    出力先：
    - `/images/visualization/heatmap`
    - `/images/visualization/edges`
    - `/images/visualization/Energy_magfield`
    - `/images/visualization/Energy_vector`

```python
import os
import sys
from glob import glob

sys.path.append(os.getcwd() + "/src")

from config.params import SNAP_PATH, datasets
from Visualization.Visualize import VisualizeMethod

dataset = set_dataset(input())

target_path = SNAP_PATH + f"/snap{dataset}"
viz = VisualizeMethod(dataset)

files = {}
files["density"] = glob(target_path + f"/density/*/*")
files["velocityx"] = glob(target_path + f"/velocityx/*/*")
files["velocityy"] = glob(target_path + f"/velocityy/*/*")
for dens_path, vx_path, vy_path in zip(files["density"], files["velocityx"], files["velocityy"]):
    viz.drawEnergy_for_velocity(dens_path, vx_path, vy_path)

files["magfieldx"] = glob(target_path + f"/magfieldx/*/*")
files["magfieldy"] = glob(target_path + f"/magfieldy/*/*")
for magx_path, magy_path in zip(files["magfieldx"], files["magfieldy"]):
    viz.drawEnergy_for_magfield(magx_path, magy_path)

files["enstrophy"] = glob(target_path + f"/enstrophy/*/*")
for target in ["velocityx", "velocityy", "magfieldx", "magfieldy", "density", "enstrophy"]:
    for path in files[target]:
        viz.drawHeatmap(path)
        viz.drawEdge(path)

```

<br>
<br>

### 3.2. AVS
処理ファイル：`/src/AVS`
出力先
- `/images/AVS/*`

<br>
<br>

### 3.3. StreamLine
処理ファイル：`/src/StreamLines`
出力先
- `/images/StreamLines/*`

<br>
<br>

### 3.4. LIC
- LIC法にて可視化する
- 縦625 * 横256 の加工済みnpyデータを使う
- snaps/half_left/snap77/magfield に保存したデータをすべて処理するために、3.9GHz, 10並列で15時間程度かかる

    処理ファイル：`/src/LIC`
    出力先
    - `/images/LIC/snap{i}/left/*`
    - `/images/LIC/snap{i}/right/*`

```python
import os
import sys
from glob import glob
sys.path.append(os.getcwd() + "/src")

from config.params import SNAP_PATH, IMAGE_PATH, datasets
from LIC.LIC import LicMethod

# ログ取得の開始
print("START", "処理開始\n\n")

dataset = set_dataset(input())
size = "left" # right

print("START", f"{dataset}.{size.split('_')[1]} 開始\n")
lic = LicMethod()

# 入出力用path の作成
in_dir = SNAP_PATH + f"/{size}/snap{dataset}"
dir_basename = os.path.basename(in_dir)  # snap77
out_dir = IMAGE_PATH + "/LIC"
base_out_path = out_dir + "/" + os.path.basename(in_dir) + "/" + size.split("_")[1]  # ./IMAGES/LIC/snap77/left
lic.makedir(f"/LIC/snap{dataset}/{size.split('_')[1]}")

# バイナリファイルの取得
binary_paths = glob(in_dir + "/magfieldx/*/*.npy")

# ファイルが無い場合
if binary_paths == []:
    print("ERROR", "File not Found\n")
    sys.exit()

for xfile in binary_paths:
    print("START", f"{os.path.splitext(os.path.basename(xfile))[0]} 開始\n")
    yfile = xfile.replace("magfieldx", "magfieldy")
    file_name = os.path.splitext(os.path.basename(xfile.replace("magfieldx", "magfield")))
    out_path = base_out_path + f"/lic_{dir_basename}.{os.path.basename(base_out_path)}.{file_name[0]}.bmp"
    # print(out_path) # ./IMAGES/LIC/snap77/lic_snap77.magfieldx.01.14.bmp

    if not os.path.exists(out_path):
        # 引数の作成
        props = lic.set_command(xfile, yfile, out_path)
        # 実行 (1画像20分程度)
        lic.LIC(props)

        # temp ファイルの削除
        lic.delete_tempfile(props[1], props[2])

    print("END", f"{os.path.splitext(os.path.basename(xfile))[0]} 終了\n")

print("END", f"{dataset} 終了\n")

```
