setlocal ENABLEDELAYEDEXPANSION
@echo off

mkdir data images logs snaps
mkdir ML\data ML\models ML\result
mkdir ML\data\LIC_labels ML\data\snpa_files
mkdir ML\models\npz ML\models\model

cd snaps

set dataset=77 497 4949
for %%d in (%dataset%) do (
	for /l %%n in (0,1,14) do (
        set num=0%%n
        set num=!num:~-2,2!

        mkdir snap%%d\density\!num!
        mkdir snap%%d\pressure\!num!
        mkdir snap%%d\velocityx\!num!
        mkdir snap%%d\velocityy\!num!
        mkdir snap%%d\velocityz\!num!
        mkdir snap%%d\magfieldx\!num!
        mkdir snap%%d\magfieldy\!num!
        mkdir snap%%d\magfieldz\!num!
        mkdir snap%%d\enstrophy\!num!
	)
)
endlocal