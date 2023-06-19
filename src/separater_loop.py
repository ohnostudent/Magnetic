import shutil
import os
import subprocess


items1 = ["density", "enstrophy", "pressure","magfieldx", "magfieldy", "magfieldz", "velocityx","velocityy", "velocityz"]
items2 = ["magfield1", "magfield2", "magfield3", "velocity1","velocity2", "velocity3"]
for file in ["../org/ft.xy.00002.00000001.0"]:
    subprocess.run([r"separator.exe", f"{file}"])
    basename = os.path.basename(file)
    para = int(basename[18:20])
    job = int(basename[21:])
    for item2 in items2:
        xyz = {1:"x",2:"y",3:"z"}
        os.rename(item2, f"{item2[:-1]}{xyz[int(item2[-1])]}")
    for item in items1:
        newname = f"{item}.{'{0:02d}'.format(para)}.{'{0:02d}'.format(job)}"
        os.rename(item, newname)
        shutil.move(newname, f'../{item}/{"{0:02d}".format(job)}/')