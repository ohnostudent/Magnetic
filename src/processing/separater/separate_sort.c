#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void byteorder8f(double *data, int num){
	int i;
	char c[10];
	char *cp;
	for(i=0;i<num;i++){
		cp = (char *)&data[i];
		c[0] = cp[0]; c[1] = cp[1]; c[2] = cp[2]; c[3]=cp[3];
		c[4] = cp[4]; c[5] = cp[5]; c[6] = cp[6]; c[7]=cp[7];

		cp[0] = c[7]; cp[1] = c[6]; cp[2] = c[5]; cp[3] = c[4];
		cp[4] = c[3]; cp[5] = c[2]; cp[6] = c[1]; cp[7] = c[0];
	}
}

void byteorder4f(float *data, int num){
	int i;
	char c[10];
	char *cp;
	for(i=0;i<num;i++){
		cp = (char *)&data[i];
		c[0] = cp[0]; c[1] = cp[1]; c[2] = cp[2]; c[3]=cp[3];
		cp[0] = c[3]; cp[1] = c[2]; cp[2] = c[1]; cp[3] = c[0];
	}
}

void byteorder4i(int *data, int num){
	int i;
	char c[10];
	char *cp;
	for(i=0;i<num;i++){
		cp = (char *)&data[i];
		c[0] = cp[0]; c[1] = cp[1]; c[2] = cp[2]; c[3]=cp[3];
		cp[0] = c[3]; cp[1] = c[2]; cp[2] = c[1]; cp[3] = c[0];
	}
}

void statis(int mx, int my, int mz, float *a, char plabel[])
{
	double amax,amin,aavr,avar,adev;

	aavr = 0.0;
	avar = 0.0;
	amax = -1.0e+24;
	amin = 1.0e+24;

	for(int k=0;k<mz;k++){
		for(int j=0;j<my;j++){
			for(int i=0;i<mx;i++){
				double val = a[i+mx*j+mx*my*k];
				if(val > amax) amax = val;
				if(val < amin) amin = val;
				aavr=aavr+val;
				avar=avar+val*val;
			}
		}
	}

	aavr = aavr/(double)(mx*my*mz);
	avar = avar/(double)(mx*my*mz);
	adev = sqrt(avar - aavr*aavr);

	printf("[statis] %s %.15E %.15E %.15E %.15E %.15E\n",plabel, amax, amin, aavr, avar, adev);
	return;
}


void ReadChar(FILE *fp, char *c, int n){
	int head, tail;
//	fread(&head, sizeof(int), 1, fp);
	fread(c, sizeof(char), n, fp);
//	fread(&tail, sizeof(int), 1, fp);
}

void Skip4Byte(FILE *fp){
	int head;
	fread(&head, sizeof(int), 1, fp);
}

void ReadInt(FILE *fp, int *inum, int n){
	int head, tail;
//	fread(&head, sizeof(int), 1, fp);
	fread(inum, sizeof(int), n, fp);
//	fread(&tail, sizeof(int), 1, fp);
	byteorder4i(inum, n);
}

void ReadDouble(FILE *fp, double *dnum, int n){
	int head, tail;
//	fread(&head, sizeof(int), 1, fp);
	fread(dnum, sizeof(double), n, fp);
//	fread(&tail, sizeof(int), 1, fp);
	byteorder8f(dnum, n);
}

void savedata(float *data, int num, char fname[]){
	FILE *fp=NULL;
	if((fp=fopen(fname, "wb"))==NULL){
		printf("file open error for write %s¥n", fname);
		exit(1);
	}

	fwrite(data, sizeof(float), num, fp);
	fclose(fp);
}

int main(int argc, char **argv)
{
	char cdocs[512+1];
	double *Vars2Print = NULL; // 3D
	double *coord1P = NULL, *coord2P = NULL, *coord3P = NULL; // 1D
	double time_loc, ax, ay, az;
	int iunit,junit,Nx,Ny,Nz,NGx1,NGy1,NGz1,NGx2,NGy2,NGz2,
	     IG_init_divX,IG_init_divY,IG_init_divZ,msz,ix,iy,iz,
	     ixsize,iysize,izsize;
	int i, j, k;
// --
	int kzwidth=3; //, junitmin, junitmax, jcount, jset=300, jskip=4, jloop=0;
	const double gamma = 5.0/3.0;
	const double gm1 = 2.0/3.0; //gamma-1.0;
	double ab1, ab2, ab3, av1, av2, av3, am1, am2, am3, aden, aprs;
	float *coord1 = NULL, *coord2 = NULL, *coord3 = NULL; // 1D
	float *var1 = NULL, *var2 = NULL, *var3 = NULL, *var4 = NULL, *var5 = NULL,
	 *var6 = NULL, *var7 = NULL, *var8 = NULL, *var9 = NULL, *var10 = NULL;//3D
	char infile[128];
	char outfile[128];
	FILE *fp = NULL;
	int loop = 0;
	int jb, njb;
	int jbmin, jbmax;
	int nfound;
// --
	jbmin = atoi(argv[1]);
	jbmax = atoi(argv[2]);
	if (argc != 3) {
		printf("usage: ./program minjobnum maxjobnum\n");
		exit(1);
	}

	for (jb = jbmin; jb < jbmax+1;jb++) {
		njb = 1;
		while (1) {
			sprintf(infile, "ft.xy.00002.%08d.%d", njb, jb);
			sprintf(cdocs, "%s", "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
			nfound = 0;
			if ((fp = fopen(infile, "rb")) == NULL) {
				printf("file open error %s\n", infile);
				nfound = 1;
				break;
			}
			Skip4Byte(fp);
			ReadInt(fp, &Nx, 1); ReadInt(fp, &Ny, 1); ReadInt(fp, &Nz, 1);
			ReadInt(fp, &NGx1, 1); ReadInt(fp, &NGy1, 1); ReadInt(fp, &NGz1, 1);
			ReadInt(fp, &NGx2, 1); ReadInt(fp, &NGy2, 1); ReadInt(fp, &NGz2, 1);
			ReadInt(fp, &IG_init_divX, 1); ReadInt(fp, &IG_init_divY, 1); ReadInt(fp, &IG_init_divZ, 1);
			ReadInt(fp, &msz, 1);
			ReadInt(fp, &ix, 1); ReadInt(fp, &iy, 1); ReadInt(fp, &iz, 1);
			ReadDouble(fp, &time_loc, 1);
			Skip4Byte(fp);

			printf("[main] params= %d %d %d %d %d %d %d %d %d %d %e\n", Nx, Ny, Nz, IG_init_divX, IG_init_divY, IG_init_divZ,
				msz, ix, iy, iz, time_loc);

			Skip4Byte(fp);
			ReadChar(fp, cdocs, 512); // 512 characters
			Skip4Byte(fp);
			printf("[main] cdocs=%s\n", cdocs);

			int xcpsize, ycpsize, zcpsize;
			xcpsize = (Nx + 1) * IG_init_divX + NGx2 + NGx1 + 1;
			ycpsize = (Ny + 1) * IG_init_divY + NGy2 + NGy1 + 1;
			zcpsize = (Nz + 1) * IG_init_divZ + NGz2 + NGz1 + 1;
			printf("Coord Size : x=%d, y=%d, z=%d\n", xcpsize, ycpsize, zcpsize);

			if (loop == 0) {
				coord1P = (double*)malloc(sizeof(double) * xcpsize);
				coord2P = (double*)malloc(sizeof(double) * ycpsize);
				coord3P = (double*)malloc(sizeof(double) * zcpsize);
			}

			Skip4Byte(fp);
			ReadDouble(fp, coord1P, xcpsize); ReadDouble(fp, coord2P, ycpsize); ReadDouble(fp, coord3P, zcpsize);
			Skip4Byte(fp);
			printf("[main] coords=%s\n", cdocs);

			Vars2Print = (double*)malloc(sizeof(double) * xcpsize * ycpsize * msz);

			Skip4Byte(fp);
			ReadDouble(fp, Vars2Print, xcpsize * ycpsize * msz);
			Skip4Byte(fp);

			fclose(fp);

			ixsize = (Nx + 1) * IG_init_divX + 1;
			iysize = (Ny + 1) * IG_init_divY + 1;
			izsize = (Nz + 1) * IG_init_divZ + 1;

			printf("[main] ixsize = %d, iysize = %d, izsize = %d\n", ixsize, iysize, izsize);

			if (loop == 0) {
				coord1 = (float*)malloc(sizeof(float) * ixsize);
				coord2 = (float*)malloc(sizeof(float) * iysize);
				coord3 = (float*)malloc(sizeof(float) * kzwidth);
			}
			for (i = 0; i < ixsize; i++) coord1[i] = coord1P[NGx1 + i];
			for (j = 0; j < iysize; j++) coord2[j] = coord2P[NGy1 + j];
			for (k = 0; k < kzwidth; k++) coord3[k] = 0.1 * k;

			if (loop == 0) {
				var1 = (float*)malloc(sizeof(float) * ixsize * iysize * kzwidth);
				var2 = (float*)malloc(sizeof(float) * ixsize * iysize * kzwidth);
				var3 = (float*)malloc(sizeof(float) * ixsize * iysize * kzwidth);
				var4 = (float*)malloc(sizeof(float) * ixsize * iysize * kzwidth);
				var5 = (float*)malloc(sizeof(float) * ixsize * iysize * kzwidth);
				var6 = (float*)malloc(sizeof(float) * ixsize * iysize * kzwidth);
				var7 = (float*)malloc(sizeof(float) * ixsize * iysize * kzwidth);
				var8 = (float*)malloc(sizeof(float) * ixsize * iysize * kzwidth);
				var9 = (float*)malloc(sizeof(float) * ixsize * iysize * kzwidth);
			}

			for (k = 0; k < kzwidth; k++) {
				for (j = 0; j <= (Ny + 1) * IG_init_divY; j++) {
					for (i = 0; i <= (Nx + 1) * IG_init_divX; i++) {
						aden = Vars2Print[i + NGx1 + (j + NGy1) * xcpsize];
						av1 = Vars2Print[i + NGx1 + (j + NGy1) * xcpsize + 2 * xcpsize * ycpsize] / aden;
						av2 = Vars2Print[i + NGx1 + (j + NGy1) * xcpsize + 3 * xcpsize * ycpsize] / aden;
						av3 = Vars2Print[i + NGx1 + (j + NGy1) * xcpsize + 4 * xcpsize * ycpsize] / aden;
						am1 = Vars2Print[i + NGx1 + (j + NGy1) * xcpsize + 5 * xcpsize * ycpsize];
						am2 = Vars2Print[i + NGx1 + (j + NGy1) * xcpsize + 6 * xcpsize * ycpsize];
						am3 = Vars2Print[i + NGx1 + (j + NGy1) * xcpsize + 7 * xcpsize * ycpsize];
						ab1 = Vars2Print[i + NGx1 + (j + NGy1) * xcpsize + 10 * xcpsize * ycpsize] + am1; // Bx
						ab2 = Vars2Print[i + NGx1 + (j + NGy1) * xcpsize + 11 * xcpsize * ycpsize] + am2; // By
						ab3 = Vars2Print[i + NGx1 + (j + NGy1) * xcpsize + 12 * xcpsize * ycpsize] + am3; // Bz
						aprs = (Vars2Print[i + NGx1 + (j + NGy1) * xcpsize + xcpsize * ycpsize]
							- 0.5 * aden * (av1 * av1 + av2 * av2 + av3 * av3) - 0.5 * (ab1 * ab1 + ab2 * ab2 + ab3 * ab3)) * gm1;

						var1[i + j * ixsize + k * ixsize * iysize] = aden;
						var2[i + j * ixsize + k * ixsize * iysize] = aprs;
						var3[i + j * ixsize + k * ixsize * iysize] = av1;
						var4[i + j * ixsize + k * ixsize * iysize] = av2;
						var5[i + j * ixsize + k * ixsize * iysize] = av3;
						var6[i + j * ixsize + k * ixsize * iysize] = ab1;
						var7[i + j * ixsize + k * ixsize * iysize] = ab2;
						var8[i + j * ixsize + k * ixsize * iysize] = ab3;
						var9[i + j * ixsize + k * ixsize * iysize] = Vars2Print[i + NGx1 + (j + NGy1) * xcpsize + 9 * xcpsize * ycpsize];
					}
				}
			}

			statis(ixsize, iysize, kzwidth, var1, "density");
			statis(ixsize, iysize, kzwidth, var2, "pressure");
			statis(ixsize, iysize, kzwidth, var3, "velocity1");
			statis(ixsize, iysize, kzwidth, var4, "velocity2");
			statis(ixsize, iysize, kzwidth, var5, "velocity3");
			statis(ixsize, iysize, kzwidth, var6, "magfield1");
			statis(ixsize, iysize, kzwidth, var7, "magfield2");
			statis(ixsize, iysize, kzwidth, var8, "magfield3");
			statis(ixsize, iysize, kzwidth, var9, "enstrophy");

			if (loop == 0) {
				savedata(coord1, ixsize, "coordx");
				savedata(coord2, iysize, "coordy");
				savedata(coord3, kzwidth, "coordz");
			}

			sprintf(outfile, "density/%02d/density.%02d.%02d", jb,njb,jb);
			savedata(var1, ixsize * iysize * kzwidth, outfile);
			sprintf(outfile, "pressure/%02d/pressure.%02d.%02d", jb,njb,jb);
			savedata(var2, ixsize * iysize * kzwidth, outfile);
			sprintf(outfile, "velocityx/%02d/velocityx.%02d.%02d", jb,njb,jb);
			savedata(var3, ixsize * iysize * kzwidth, outfile);
			sprintf(outfile, "velocityy/%02d/velocityy.%02d.%02d", jb,njb,jb);
			savedata(var4, ixsize * iysize * kzwidth, outfile);
			sprintf(outfile, "velocityz/%02d/velocityz.%02d.%02d", jb,njb,jb);
			savedata(var5, ixsize * iysize * kzwidth, outfile);
			sprintf(outfile, "magfieldx/%02d/magfieldx.%02d.%02d", jb,njb,jb);
			savedata(var6, ixsize * iysize * kzwidth, outfile);
			sprintf(outfile, "magfieldy/%02d/magfieldy.%02d.%02d", jb,njb,jb);
			savedata(var7, ixsize * iysize * kzwidth, outfile);
			sprintf(outfile, "magfieldz/%02d/magfieldz.%02d.%02d", jb,njb,jb);
			savedata(var8, ixsize * iysize * kzwidth, outfile);
			sprintf(outfile, "enstrophy/%02d/enstrophy.%02d.%02d", jb,njb,jb);
			savedata(var9, ixsize * iysize * kzwidth, outfile);

			// sprintf(outfile, "density%08d", loop);
			// savedata(var1, ixsize * iysize * kzwidth, outfile);
			// sprintf(outfile, "pressure%08d", loop);
			// savedata(var2, ixsize * iysize * kzwidth, outfile);
			// sprintf(outfile, "velocityx%08d", loop);
			// savedata(var3, ixsize * iysize * kzwidth, outfile);
			// sprintf(outfile, "velocityy%08d", loop);
			// savedata(var4, ixsize * iysize * kzwidth, outfile);
			// sprintf(outfile, "velocityz%08d", loop);
			// savedata(var5, ixsize * iysize * kzwidth, outfile);
			// sprintf(outfile, "magfieldx%08d", loop);
			// savedata(var6, ixsize * iysize * kzwidth, outfile);
			// sprintf(outfile, "magfieldy%08d", loop);
			// savedata(var7, ixsize * iysize * kzwidth, outfile);
			// sprintf(outfile, "magfieldz%08d", loop);
			// savedata(var8, ixsize * iysize * kzwidth, outfile);
			// sprintf(outfile, "enstrophy%08d", loop);
			// savedata(var9, ixsize * iysize * kzwidth, outfile);

			/*
			!  columns for  NS3DComp,        XMHD3DComp::
			!    1-2        (i,j)            (i,j)
			!    3-5        (x,y,z)         (x,y,z)
			!    6-10       den,enet,amo1-3  den,enet,amo1-3
			!    11         dilata           mag1
			!    12         enstro           mag2
			!    13                          mag3
			!    14                          B0magX
			!    15                          B0magY
			!    16                          B0magZ
			!    17                          pres0
			!    18                          dilata
			!    19                          enstro
			*/
			if (loop == 0) {
				printf("[main] coord1 ");
				for (i = 0; i < ixsize; i++) {
					printf("%f ", coord1[i]);
				}
				printf("\n");

				printf("[main] coord2 ");
				for (j = 0; j < iysize; j++) {
					printf("%f ", coord2[j]);
				}
				printf("\n");

				printf("[main] coord3 ");
				for (k = 0; k < kzwidth; k++) {
					printf("%f ", coord3[j]);
				}
				printf("\n");
			}
			njb++;
			loop++;
		}
		if(njb == 1 && nfound == 1) exit(1);
	}
	free(coord1P); free(coord2P); free(coord3P);
	free(coord1); free(coord2); free(coord3);
	free(var1); free(var2); free(var3); free(var4); free(var5);
	free(var6); free(var7); free(var8); free(var9);

	return 0;
}


