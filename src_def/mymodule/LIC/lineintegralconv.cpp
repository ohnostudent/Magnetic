#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <random>

#include "lineintegralconv.h"

LineIntegralConv :: LineIntegralConv(){
  axis_min[0] = 0.0;
  axis_min[1] = 0.0;
  numlicpxl = 0;
  debug = false;
}

void LineIntegralConv :: setDebug(bool db)
{
 debug = db;
}

void LineIntegralConv :: writeBmp(char *filename)
{
  FILE *fp;
  int i, j, k, l, m;
  int filesize, width, height, cwidth;
  int val = 255;
  const int MaxC=255;
  const int OFFSET = 54;
  char val1, val2, val3, val4;
  char *imageUC;
  int red, green, blue;

  width = pic_w;  height = pic_h;
  filesize = (width*3 + width%4)*height+OFFSET;
  //imageUC = (char *)malloc(filesize);
  imageUC = new char[filesize];
  /* BMP Header*/
  /* ---------File Header --------- */
     imageUC[0] = 'B'; imageUC[1] = 'M'; 
     /* File size 4 bytes*/
     val1 = (char)(val&filesize); val2 = (char)(val&(filesize>>8));
     val3 = (char)(val & (filesize>>16));  val4 = (char)(val&(filesize>>24));

    imageUC[2] = val1;  imageUC[3] = val2;
    imageUC[4] = val3;  imageUC[5] = val4;

    for(i=6;i<=9;i++) imageUC[i] = (char)0; /* reserved area*/

    /* offset to image data*/
     imageUC[10] = (char)54; imageUC[11] = (char)0;
     imageUC[12] = (char)0; imageUC[13] = (char)0;
    /* --------- Information Header ---------*/
    /* information header size */
     imageUC[14] = (char)40; imageUC[15] = (char)0;
     imageUC[16] = (char)0; imageUC[17] = (char)0;
    /* width size */
     val1 = (char)(val&width); val2 = (char)(val&(width >>8));
     val3 = (char)(val &(width >>16));  val4 = (char)(val&(width>>24));

    imageUC[18] = val1;  imageUC[19] = val2;
    imageUC[20] = val3;  imageUC[21] = val4;

     /* Height size*/
     val1 = (char)(val&height); val2 = (char)(val&(height>>8));
     val3 = (char)(val&(height>>16));  val4 = (char)(val&(height>>24));

     imageUC[22] = val1;  imageUC[23] = val2;
     imageUC[24] = val3;  imageUC[25] = val4;

     imageUC[26] = (char)1; imageUC[27] = (char)0; 
     imageUC[28] = (char)24; imageUC[29] = (char)0; 
     imageUC[30] = (char)0; imageUC[31] = (char)0; 
     imageUC[32] = (char)0; imageUC[33] = (char)0;
    /*image data size*/
     imageUC[34] = (char)0; imageUC[35] = (char)0;
     imageUC[36] = (char)0; imageUC[37] = (char)0;

     for(i=38;i<=53;i++) imageUC[i] = (char)0; /* reserved area*/

    cwidth = 3*width + (width%4);

    for(j=0;j<height;j++){
      for(i=0;i<width;i++){
         l = 3*i+j*cwidth+OFFSET;
         m = 3*(i+j*width);
         red = 256*rgb[m];
         green = 256*rgb[m+1];
         blue = 256*rgb[m+2];
         if(red < 0) red = 0;
         if(red > 255) red = 255;

         if(green < 0) green = 0;
         if(green > 255) green = 255;

         if(blue < 0) blue = 0;
         if(blue > 255) blue = 255;

         imageUC[l] = (unsigned char)red;
         imageUC[l+1] = (unsigned char)green;
         imageUC[l+2] = (unsigned char)blue;
      }
    }

  if ((fp=fopen(filename, "wb"))==NULL){
    printf("File Open for png failed\n");
    exit(1);
  }
  fwrite(imageUC, 1, filesize, fp);
  fclose(fp);

  delete [] imageUC;
  return;
}


void LineIntegralConv :: loadData(char *filex, char *filey){
  FILE *fp;
  if ((fp=fopen(filex, "rb"))==NULL){
    printf("File Open for fx (%s)  failed\n",filex);
    exit(1);
  }
  fread(fx, sizeof(float), dat_size[0]*dat_size[1], fp);
  fclose(fp);

  if ((fp=fopen(filey, "rb"))==NULL){
    printf("File Open for fy (%s) failed\n",filey);
    exit(1);
  }
  fread(fy, sizeof(float), dat_size[0]*dat_size[1], fp);
  fclose(fp);

}

void LineIntegralConv :: setDatsize(int x, int y){
  dat_size[0] = x;
  dat_size[1] = y;
  if(debug) printf("datasize = %d, %d\n",dat_size[0], dat_size[1]);
}

void LineIntegralConv :: setLen(int l){
  numlicpxl = l;
  if(numlicpxl*40 > rungekutta_step) 
    rungekutta_step = numlicpxl*40;
}


void LineIntegralConv :: setAxisMin(double x, double y){
  axis_min[0] = x;
  axis_min[1] = y;
}
void LineIntegralConv :: setDelta(double dx, double dy){
  axis_delta[0] = dx;
  axis_delta[1] = dy;
  if(debug) printf("delta = %f, %f\n",axis_delta[0],axis_delta[1]);

  region_min[0] = axis_min[0];
  region_min[1] = axis_min[1];
  if(debug) printf("regionmin = %f, %f\n",region_min[0], region_min[1]);

  region_max[0] = axis_min[0] + (dat_size[0]-1)*axis_delta[0];
  region_max[1] = axis_min[1] + (dat_size[1]-1)*axis_delta[1];
  if(debug) printf("regionmax = %f, %f\n",region_max[0], region_max[1]);
}

void LineIntegralConv :: setwh(int x, int y, int m){
  mag = m;
  pic_w = x*mag;
  pic_h = y*mag*axis_delta[1]/axis_delta[0];

  if(debug) printf("pic width, height = %d, %d\n",pic_w, pic_h);
  if(debug) printf("pic mag = %d\n",mag);

  mag_delta[0] = (region_max[0]-region_min[0])/(pic_w-1);
  mag_delta[1] = (region_max[1]-region_min[1])/(pic_h-1);
  
  if(debug) printf("pic mag delta= %f, %f\n",mag_delta[0], mag_delta[1]);
}

void LineIntegralConv :: setStreamParam(double div, int rs){
 stream_div = div;
 rungekutta_step = rs;
  if(debug) printf("LineIntegralConv, div, rungekutta max step = %f, %d\n",stream_div, rungekutta_step);
}

void LineIntegralConv :: allocMem(){
 if(debug) printf("allocMem numlicpxl=%d\n",numlicpxl);
 int x = dat_size[0];
 int y = dat_size[1];
 fx = new float[x*y];
 fy = new float[x*y];
 rgb = new float[pic_w*pic_h*3];
 whitenoise = new float[pic_w*pic_h];
 for(int j=0;j<2;j++)
  for(int i=0;i<2;i++)
    licpxl[j][i] = new int [numlicpxl];
}

void LineIntegralConv :: freeMem(){
 if(debug) printf("freeMem\n");
 delete [] fx; delete [] fy;
 delete [] rgb; delete [] whitenoise;
 for(int j=0;j<2;j++)
  for(int i=0;i<2;i++)
   delete [] licpxl[j][i];

}

void LineIntegralConv :: makeLIC(){
 int sp[2];
 double pos[2];
 float c;
 for (int j = 0; j < pic_h; j++)
   {
     for (int i = 0; i < pic_w; i++)
       {
         sp[0] = i; sp[1] = j;
         pxl2pos(sp, pos);
         //printf("%f,%f\n",pos[0],pos[1]);
         c = stream_line(sp);
	 rgb[(i + j * pic_w) * 3] = c;
	 rgb[(i + j * pic_w) * 3 + 1] = c;
	 rgb[(i + j * pic_w) * 3 + 2] = c;
	 //printf("c = %f\n",c);
       }
       if(debug) printf("%d/%d\n",j,pic_h);
   }
   if(debug) printf("makeLic ends\n");
}

/*

 ***
 Runge Kutta and Interpolation ware replaced by VFIVE's subroutines.
 ***

 stream_line(double x0[3], float color0[4])

 x0[3]		: start point
 color0[4]	: mono(color of line)
 */

static float
vect_inner_product (float a[], float b[])
{
  return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}


static double
vect_amp(float vec[])
{
  double vx, vy, vz;
  vx = vec[0];
  vy = vec[1];
  vz = vec[2];
  return (sqrt (vx * vx + vy * vy + vz * vz));
}


void
LineIntegralConv::pos2pxl (double *pos, int *pxl)
{
  pxl[0] = (pos[0]-region_min[0])/mag_delta[0];
  pxl[1] = (pos[1]-region_min[1])/mag_delta[1];
}

void
LineIntegralConv::pxl2pos (int *pxl, double *pos)
{
  pos[0] = (region_min[0] + pxl[0]*mag_delta[0]);
  pos[1] = (region_min[1] + pxl[1]*mag_delta[1]);
}
  
void
LineIntegralConv::preproc ()
{
  random_device rnd;
  mt19937 mt(rnd());
  uniform_int_distribution<int> intRand(0, 255);
  for (int j = 0; j < pic_h; j++)
    {
      for (int i = 0; i < pic_w; i++)
	{
	  rgb[(i + j * pic_w) * 3] = 0.0;
	  rgb[(i + j * pic_w) * 3 + 1] = 0.0;
	  rgb[(i + j * pic_w) * 3 + 2] = 0.0;
	  
	  whitenoise[i + j * pic_w] = intRand(mt)/(float)255;
	  //if(debug)printf("%f ",whitenoise[i + j * pic_w]);
	}
    }
    if(debug) printf("\n");

}


int LineIntegralConv:: find_fact (double x, double y, int *i, int *j, double cx[4], double cy[4])
{
  double vii, vjj;
  double hx, hy;

  int n1 = dat_size[0];
  int n2 = dat_size[1];

  if (x <= region_min[0] || x >= region_max[0]
      || y <= region_min[1] || y >= region_max[1])
    return (0);

  vii = (x - axis_min[0]) / axis_delta[0];
  *i = (int) vii;
  hx = vii - *i;

  vjj = (y - axis_min[1]) / axis_delta[1];
  *j = (int) vjj;
  hy = vjj - *j;

  if (*i < 0 || *i > n1 - 2 || *j < 0 || *j > n2 - 2) return (0);

  // debug s
  if (hx < 0.0 || hx > 1.0) printf ("  ** error.  hx = %f\n", hx);
  if (hy < 0.0 || hy > 1.0) printf ("  ** error.  hy = %f\n", hy);
  // debug e

  if (*i == 0 || *i == n1 - 2)
    {
      cx[0] = (1.0 - hx) / 2.0;
      cx[1] = (1.0 - hx) / 2.0;
      cx[2] = hx / 2.0;
      cx[3] = hx / 2.0;
    }
  else
    {
      cx[0] = -hx * (2.0 - hx) * (1.0 - hx) / 6.0;
      cx[1] = (1.0 + hx) * (2.0 - hx) * (1.0 - hx) / 2.0;
      cx[2] = hx * (2.0 - hx) * (1.0 + hx) / 2.0;
      cx[3] = -hx * (1.0 - hx) * (1.0 + hx) / 6.0;
    }

  if (*j == 0 || *j == n2 - 2)
    {
      cy[0] = (1.0 - hy) / 2.0;
      cy[1] = (1.0 - hy) / 2.0;
      cy[2] = hy / 2.0;
      cy[3] = hy / 2.0;
    }
  else
    {
      cy[0] = -hy * (2.0 - hy) * (1.0 - hy) / 6.0;
      cy[1] = (1.0 + hy) * (2.0 - hy) * (1.0 - hy) / 2.0;
      cy[2] = hy * (2.0 - hy) * (1.0 + hy) / 2.0;
      cy[3] = -hy * (1.0 - hy) * (1.0 + hy) / 6.0;
    }

  return (1);
}

double
LineIntegralConv::interpolator (int i, int j, double *cx, double *cy,	int ele)
{
  double b01, b02, b03, b04;
  double b17;
  int ip1, jp1, im1, jm1, ip2, jp2;
  int n1, n2;

  float *q;

  n1 = dat_size[0];
  n2 = dat_size[1];

  im1 = i - 1;
  jm1 = j - 1;
  ip1 = i + 1;
  ip2 = i + 2;
  jp1 = j + 1;
  jp2 = j + 2;

  if (i == 0)
    {
      im1 = i;
      ip2 = ip1;
    }

  if (j == 0)
    {
      jm1 = j;
      jp2 = jp1;
    }

  if (i == n1 - 2)
    {
      ip2 = ip1;
      im1 = i;
    }

  if (j == n2 - 2)
    {
      jp2 = jp1;
      jm1 = j;
    }

  if(ele == 0) q = fx;
  	else q = fy;

  b01 = cx[0] * q[jm1 * n1 + im1] +
        cx[1] * q[jm1 * n1 + i] + 
        cx[2] * q[jm1 * n1 + ip1] +
        cx[3] * q[jm1 * n1 + ip2];


  b02 = cx[0] * q[j * n1 + im1] + 
        cx[1] * q[j * n1 + i] + 
        cx[2] * q[j * n1 + ip1] +
        cx[3] * q[j * n1 + ip2];

  b03 = cx[0] * q[jp1 * n1 + im1] +
        cx[1] * q[jp1 * n1 + i] + 
        cx[2] * q[jp1 * n1 + ip1] + 
        cx[3] * q[jp1 * n1 + ip2];

  b04 = cx[0] * q[jp2 * n1 + im1] + 
        cx[1] * q[jp2 * n1 + i] + 
        cx[2] * q[jp2 * n1 + ip1] + 
        cx[3] * q[jp2 * n1 + ip2];

  //if(debug) printf("%f, %f, %f, %f\n",b01,b02,b03,b04);

  b17 = cy[0] * b01 + cy[1] * b02 + cy[2] * b03 + cy[3] * b04;
  //if(debug) printf("b17: %f\n",b17);

  return b17;
}



int
LineIntegralConv::vec_value (double x, double y, double z, double *vx, double *vy, double *vz, double *vec_abs)
{
  double cx[4], cy[4];
  int i, j;

  double vec_x, vec_y, vec_absvalue;

  int iout = find_fact (x, y, &i, &j, cx, cy);

  //if(debug) printf("%f, %f %d, %d\n",x,y,i,j);

  if (iout == 0) return (0);

  vec_x = interpolator (i, j, cx, cy, 0);
  vec_y = interpolator (i, j, cx, cy, 1);

  vec_absvalue = sqrt(vec_x * vec_x + vec_y * vec_y);

//  if(debug) printf("%f, %f (%f)\n",vec_x, vec_y, vec_absvalue);

  if (vec_absvalue > 1.0E-10)
    {
      *vx = vec_x / vec_absvalue;
      *vy = vec_y / vec_absvalue;
      *vz = 0.0;
      *vec_abs = vec_absvalue;
      return (1);
    }
  else
    {
      *vx = 0.0;
      *vy = 0.0;
      *vz = 0.0;
      *vec_abs = 0.0;
      return (0);
    }
}


float
LineIntegralConv:: stream_line(int x0[2])
{
  int i, j, direction;
  float h;
  double x[3];
  double vx, vy, absv;
  int if_continue;
  const int n = 3;			// 3 dim
  double f[30];
  double c01, c02, c03, c04, c05, c06, c07, c08, c09, c10;
  double c11, c12, c13, c14, c15, c16, c17, c18, c19, c20;
  double c21, c22, c23, c24, c25, c26, c27;
  int prev_x[2];
  int cpxl[2]; // current pixel
  rnlicpxl[0]=0; rnlicpxl[1]=0;

  for (direction = 0; direction < 2; direction++)
    {
      if (direction == 0) h = stream_div;
      else h = -stream_div;
      
      rnlicpxl[direction]++;
      licpxl[direction][0][0] = x0[0];
      licpxl[direction][1][0] = x0[1];
      prev_x[0] = x0[0]; prev_x[1] = x0[1];
      pxl2pos(x0, x);
      f[0] = x[0]; f[1] = x[1];

      // Set coefficients
      c01 = h / 9.0;
      c02 = h * 0.4166666666666667e-1;
      c03 = h * 0.125;
      c04 = h / 6.0;
      c05 = h * 0.5;
      c06 = h * 0.6666666666666667;
      c07 = h / 3.0;
      c08 = h * 0.3750;
      c09 = h * 0.1333333333333333e1;
      c10 = h * 0.3333333333333333e1;
      c11 = h * 0.7e1;
      c12 = h * 0.9666666666666667e1;
      c13 = h * 0.1533333333333333e2;
      c14 = h * 0.6111111111111111;
      c15 = h * 0.1166666666666667e1;
      c16 = h * 0.1375e1;
      c17 = h * 0.8333333333333333;
      c18 = h * 0.4390243902439024;
      c19 = h * 0.8780487804878049;
      c20 = h * 0.1304878048780488e1;
      c21 = h * 0.2097560975609756e1;
      c22 = h * 0.2963414634146341e1;
      c23 = h * 0.4317073170731707e1;
      c24 = h * 0.3214285714285714e-1;
      c25 = h * 0.4880952380952381e-1;
      c26 = h * 0.2571428571428571;
      c27 = h * 0.3238095238095238;


      //project_lines2plane (x, line_color);

      for (j = 1; j < rungekutta_step; j++)
	{

	  // Calculations of 6th 8stages Runge Kutta

	  /* 1st stage */
	  if_continue = vec_value (x[0], x[1], x[2], &f[2 * n], &f[2 * n + 1], &f[2 * n + 2], &absv);

	  if (if_continue == 0) break;

	  for (i = 0; i < n; i++)
	    {
	      f[n + i] = c01 * f[2 * n + i] + f[i];
	    }

	  /* 2nd stage */
	  if_continue = vec_value (f[n], f[n + 1], f[n + 2], &f[3 * n], &f[3 * n + 1], &f[3 * n + 2], &absv);

	  if (if_continue == 0)
	    break;

	  for (i = 0; i < n; i++)
	    {
	      f[n + i] = c02 * f[2 * n + i] + c03 * f[3 * n + i] + f[i];
	    }

	  /* 3rd stage */
	  if_continue = vec_value (f[n], f[n + 1], f[n + 2], &f[4 * n], &f[4 * n + 1], &f[4 * n + 2], &absv);

	  if (if_continue == 0)
	    break;

	  for (i = 0; i < n; i++)
	    {

	      f[n + i] = c04 * f[2 * n + i] - c05 * f[3 * n + i] + c06 * f[4 * n + i] + f[i];

	    }

	  /* 4th stage */
	  if_continue = vec_value (f[n], f[n + 1], f[n + 2], &f[5 * n], &f[5 * n + 1], &f[5 * n + 2], &absv);

	  if (if_continue == 0)
	    break;

	  for (i = 0; i < n; i++)
	    {
	      f[n + i] = c03 * f[2 * n + i] + c08 * f[5 * n + i] + f[i];
	    }

	  /* 5th stage */
	  if_continue = vec_value (f[n], f[n + 1], f[n + 2], &f[6 * n], &f[6 * n + 1], &f[6 * n + 2], &absv);

	  if (if_continue == 0) break;

	  for (i = 0; i < n; i++)
	    {
	      f[n + i] = -c09 * f[2 * n + i] + c10 * f[6 * n + i] - c11 * f[3 * n + i] - c12 * f[5 * n + i] + c13 * f[4 * n + i] + f[i];
	    }

	  /* 6th stage */
	  if_continue = vec_value (f[n], f[n + 1], f[n + 2], &f[7 * n], &f[7 * n + 1], &f[7 * n + 2], &absv);

	  if (if_continue == 0) break;

	  for (i = 0; i < n; i++)
	    {
	      f[n + i] = -c01 * f[2 * n + i] + c03 * f[7 * n + i]
		+ c14 * f[6 * n + i] - c15 * f[4 * n + i]
		+ c16 * f[3 * n + i] + f[i];
	    }

	  /* 7th stage */
	  if_continue = vec_value (f[n], f[n + 1], f[n + 2], &f[8 * n], &f[8 * n + 1], &f[8 * n + 2], &absv);

	  if (if_continue == 0) break;

	  for (i = 0; i < n; i++)
	    {
	      f[n + i] = -c18 * f[7 * n + i] + c19 * f[8 * n + i]
		+ c20 * f[2 * n + i] - c21 * f[6 * n + i]
		- c22 * f[3 * n + i] + c23 * f[5 * n + i] + f[i];
	    }

	  /* 8th stage */
	  if_continue = vec_value (f[n], f[n + 1], f[n + 2], &f[3 * n], &f[3 * n + 1], &f[3 * n + 2], &absv);

	  if (if_continue == 0) break;

	  for (i = 0; i < n; i++)
	    {
	      x[i] = f[i] = (f[5 * n + i] + f[7 * n + i]) * c24
		+ (f[2 * n + i] + f[3 * n + i]) * c25
		+ (f[4 * n + i] + f[8 * n + i]) * c26
		+ f[6 * n + i] * c27 + f[i];
	    }

          pos2pxl(x, cpxl);
          if(cpxl[0]!=prev_x[0] || cpxl[1]!=prev_x[1]){
            prev_x[0] = cpxl[0]; prev_x[1] = cpxl[1];
            licpxl[direction][0][rnlicpxl[direction]] = cpxl[0]; 
            licpxl[direction][1][rnlicpxl[direction]] = cpxl[1];
            rnlicpxl[direction]++;
            if(rnlicpxl[direction] > numlicpxl) break; 
          }
	  // End of Calculations
	  //project_lines2plane (x, line_color);
	}			// j
    }// direction
    float pxcolor = 0.0f;
    //printf("(%d,%d):%d, %d \n",x0[0],x0[1],rnlicpxl[0],rnlicpxl[1]);
    for(int dir=0;dir<2;dir++)
      for(int i=0;i<rnlicpxl[dir];i++){
        int px, py;
        px = licpxl[dir][0][i];
        py = licpxl[dir][1][i];
        //printf("color = %f, %f, %f\n",sin(PI*(i+numlicpxl)/(2*numlicpxl)),whitenoise[px+py*pic_w],sin(PI*(i+numlicpxl)/(2*numlicpxl))*whitenoise[px+py*pic_w]);
        pxcolor += sin(PI*(i+numlicpxl)/(2*numlicpxl))*whitenoise[px+py*pic_w]*(PI/(2*numlicpxl))/2.0; 
      }
    //printf("color = %f\n",pxcolor);
    return pxcolor;

}

