setlocal

set name=%1

if %name% == 4949 (
    set target=snap4949
) else if %name% == 77 (
    set target=snap77
) else (
    set target=snap497
)

mkdir .\snaps\%target%
cd .\snaps\%target%

mkdir density
mkdir density\00
mkdir density\01
mkdir density\02
mkdir density\03
mkdir density\04
mkdir density\05
mkdir density\06
mkdir density\07
mkdir density\08
mkdir density\09
mkdir density\10
mkdir density\11
mkdir density\12
mkdir density\13
mkdir density\14
mkdir pressure
mkdir pressure\00
mkdir pressure\01
mkdir pressure\02
mkdir pressure\03
mkdir pressure\04
mkdir pressure\05
mkdir pressure\06
mkdir pressure\07
mkdir pressure\08
mkdir pressure\09
mkdir pressure\10
mkdir pressure\11
mkdir pressure\12
mkdir pressure\13
mkdir pressure\14
mkdir velocityx
mkdir velocityx\00
mkdir velocityx\01
mkdir velocityx\02
mkdir velocityx\03
mkdir velocityx\04
mkdir velocityx\05
mkdir velocityx\06
mkdir velocityx\07
mkdir velocityx\08
mkdir velocityx\09
mkdir velocityx\10
mkdir velocityx\11
mkdir velocityx\12
mkdir velocityx\13
mkdir velocityx\14
mkdir velocityy
mkdir velocityy\00
mkdir velocityy\01
mkdir velocityy\02
mkdir velocityy\03
mkdir velocityy\04
mkdir velocityy\05
mkdir velocityy\06
mkdir velocityy\07
mkdir velocityy\08
mkdir velocityy\09
mkdir velocityy\10
mkdir velocityy\11
mkdir velocityy\12
mkdir velocityy\13
mkdir velocityy\14
mkdir velocityz
mkdir velocityz\00
mkdir velocityz\01
mkdir velocityz\02
mkdir velocityz\03
mkdir velocityz\04
mkdir velocityz\05
mkdir velocityz\06
mkdir velocityz\07
mkdir velocityz\08
mkdir velocityz\09
mkdir velocityz\10
mkdir velocityz\11
mkdir velocityz\12
mkdir velocityz\13
mkdir velocityz\14
mkdir magfieldx
mkdir magfieldx\00
mkdir magfieldx\01
mkdir magfieldx\02
mkdir magfieldx\03
mkdir magfieldx\04
mkdir magfieldx\05
mkdir magfieldx\06
mkdir magfieldx\07
mkdir magfieldx\08
mkdir magfieldx\09
mkdir magfieldx\10
mkdir magfieldx\11
mkdir magfieldx\12
mkdir magfieldx\13
mkdir magfieldx\14
mkdir magfieldy
mkdir magfieldy\00
mkdir magfieldy\01
mkdir magfieldy\02
mkdir magfieldy\03
mkdir magfieldy\04
mkdir magfieldy\05
mkdir magfieldy\06
mkdir magfieldy\07
mkdir magfieldy\08
mkdir magfieldy\09
mkdir magfieldy\10
mkdir magfieldy\11
mkdir magfieldy\12
mkdir magfieldy\13
mkdir magfieldy\14
mkdir magfieldz
mkdir magfieldz\00
mkdir magfieldz\01
mkdir magfieldz\02
mkdir magfieldz\03
mkdir magfieldz\04
mkdir magfieldz\05
mkdir magfieldz\06
mkdir magfieldz\07
mkdir magfieldz\08
mkdir magfieldz\09
mkdir magfieldz\10
mkdir magfieldz\11
mkdir magfieldz\12
mkdir magfieldz\13
mkdir magfieldz\14
mkdir enstrophy
mkdir enstrophy\00
mkdir enstrophy\01
mkdir enstrophy\02
mkdir enstrophy\03
mkdir enstrophy\04
mkdir enstrophy\05
mkdir enstrophy\06
mkdir enstrophy\07
mkdir enstrophy\08
mkdir enstrophy\09
mkdir enstrophy\10
mkdir enstrophy\11
mkdir enstrophy\12
mkdir enstrophy\13
mkdir enstrophy\14

endlocal