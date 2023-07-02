# -*- coding: utf-8 -*-

from glob import glob


def sort_paths(pathlist, paraloc=[-9,-8], jobloc=[-6,-5]):
    """
    pathlistはglob等で取得したlist。
    locはそれぞれのpath上での位置。
    pathlistをpara,jobでソートして返す。
    """
    pjp = [{"path": path,
        "job": int("".join([path[i] for i in jobloc])),
        "para": int("".join([path[i] for i in paraloc]))} for path in pathlist
    ]
    pjp2 = sorted(pjp, key = lambda x: (x["para"]))
    pjp3 = sorted(pjp2, key = lambda x: (x["job"]))

    return [x["path"] for x in pjp3]


def make_viewer():
    dataset = input("which dataset:")
    outname = f"lic_viewer{dataset}.html"
    paths = glob(f".\snap{dataset}\*bmp")
    paths = sort_paths(paths) #この関数はsnapの命名規則をもとに時系列順に並び変える。

    with open("viewer_template.html", "r", encoding="utf8") as f:
        html = f.read()

    pathliststr = ""
    for p in paths:
        pathliststr += f"'{p}',\n"

    html = html.replace("{replaceblock}", pathliststr)

    with open(outname, "w", encoding="utf8") as f:
        f.write(html)


if __name__ == '__main__':
    make_viewer()