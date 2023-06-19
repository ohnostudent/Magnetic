import shutil
import os
import subprocess
import glob


root_dir = os.getcwd() + "\\data"
out_dir = os.getcwd() + "\\snaps"
items1 = ["density", "enstrophy", "pressure","magfieldx", "magfieldy", "magfieldz", "velocityx","velocityy", "velocityz"]
items2 = ["magfield1", "magfield2", "magfield3", "velocity1","velocity2", "velocity3"]
xyz = {1:"x",2:"y",3:"z"}


for targ in [49, 77, 497]:
    if targ == 49:
        i, j = 49, 49
    elif targ == 77:
        i, j = 7, 7
    elif targ == 497:
        i, j = 49, 7
    else:
        raise "Value Error"

    subprocess.run([out_dir + "\\mkdirs.bat", str(targ)])
    files = glob.glob(root_dir + "\\*\\ICh.target=50.ares=1.0d-{i}.adiffArt=1.0d-{j}.h00.g00.BCv1=0.0\\Snapshots\\*".format(i=i, j=j))

    for file in files:
        subprocess.run([root_dir + "\\..\\cln\\separator.exe", f"{file}"])
        _, _, _, para, job = os.path.basename(file).split(".")
        para = int(para)
        job = int(job)
        f = open(out_dir+'\\myfile.txt', 'w')

        for item2 in items2:
            try:
                os.rename(item2, f"{item2[:-1]}{xyz[int(item2[-1])]}")
            except FileNotFoundError:
                f.write(f"Filenot Found: {item2}.{'{0:02d}'.format(para)}.{'{0:02d}'.format(job)}\n")

        f.write("\n")

        for item in items1:
            try:
                newname = f"{item}.{'{0:02d}'.format(para)}.{'{0:02d}'.format(job)}"
                os.rename(item, newname)
                shutil.move(newname, out_dir+f'/{item}/{"{0:02d}".format(job)}/')
            except FileNotFoundError:
                f.write(f"Filenot Found: {item2}.{'{0:02d}'.format(para)}.{'{0:02d}'.format(job)}\n")
        f.close()
        