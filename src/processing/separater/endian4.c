#include <stdio.h>
#include <stdlib.h>

float *fdata;

void byteorder(float *data, int num){
	int i;
	char c[10];
	char *cp;
	for(i=0;i<num;i++){
		cp = (char *)&data[i];
		c[0] = cp[0]; c[1] = cp[1]; c[2] = cp[2]; c[3]=cp[3];
		cp[0] = c[3]; cp[1] = c[2]; cp[3] = c[1]; cp[3] = c[0];
	}
}

int main(int argc, char *argv[]){
	FILE *fp = NULL;
	size_t size;
	int i, num;
	float min, max, temp;

	if(argc != 4){
		printf("./program filename outfile size\n");
		exit(1);
	} else {
		printf("infile =  %s\n",argv[1]);
		printf("outfile =  %s\n",argv[2]);
		size = atol(argv[3]);
		printf("size =  %ld (argv[3]=%s)\n",size,argv[3]);
	}
	
	fdata = (float *)malloc(size);

	fp = fopen(argv[1], "rb");
	if(fp==NULL){
		printf("File Open Error %s\n", argv[1]);
		exit(1);
	}
	
	num=fread(fdata, sizeof(float), size/sizeof(float), fp);

	fclose(fp);

	if(num*sizeof(float) != size){
		printf("num=%d(%ld)\n",num,size/sizeof(float));
		exit(1);
	}
	
	byteorder(fdata, num);

	min = max = fdata[1];
	for(i=0;i<num;i++){ // if skip header and footer -> for(i=1;i<num-1;i++){
		temp = fdata[i];
		if(fdata[i] > max) max = fdata[i];
		if(fdata[i] < min) min = fdata[i];
	}
	printf("min = %f, max = %f\n",min,max);

	fp= fopen(argv[2], "wb");
	fwrite(fdata, sizeof(float), num, fp);
	fclose(fp);
	
	free(fdata);

	return 0;
}

