# ../imgout/ohnolicのビューワーを作成するプログラム
#bmpが入っているディレクトリの親ディレクトリで実行してください
import glob
from mymodule import myfunc as mf
def main():
    dataset = input("which dataset:")
    outname = f"lic_viewer{dataset}.html"
    paths = glob.glob(f"./snap{dataset}/*bmp")
    paths = mf.sort_paths(paths) #この関数はsnapの命名規則をもとに時系列順に並び変える。
    with open("viewer_template.html", "r", encoding="utf8") as f:
        html = f.read()
    pathliststr = ""
    for p in paths:
        tempstr = p.replace("\\","/")
        pathliststr += f"'{tempstr}',\n"
    html = html.replace("{replaceblock}", pathliststr)
    with open(outname, "w", encoding="utf8") as f:
        f.write(html)
        print(len(html))

if __name__ == '__main__':
    main()