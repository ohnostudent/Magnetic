#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include "streamLines.h"

using namespace std;

const double PI = 3.1415926535897932;

// const double gridx = 513;
// const double gridy = 1025;
const double lengx = 2.0*PI;
const double lengy = 2.0;

// const double dx = lengx/(gridx-1);
// const double dy = lengy/(gridy-1);

int mag = 6;

StreamLine *streamlines;

int main(int argc, char **argv){
//  char xfile[] = "magfield1";
//  char yfile[] = "magfield2";
//  char picfile[] = "test.bmp";
 if (argc <= 3)
 {  return 0;}
char *xfile = argv[1];
char *yfile = argv[2];
char *picfile = argv[3];
 int gridx, gridy;
 if (argc == 6)
 {
  gridx = stoi(std::string(argv[4]));
  gridy = stoi(std::string(argv[5]));
 }
 else{
 gridx = 513; gridy=1025;
 }
 double dx = lengx/(gridx-1);
 double dy = lengy/(gridy-1);
 
 
 
 //-----
 streamlines = new StreamLine;
 streamlines->setDebug(true);
 streamlines->setDatsize(gridx, gridy);
 streamlines->setDelta(dx, dy);
  // mag decide out image size.
 streamlines->setwh(gridx, gridy, mag);
 double div = dy*0.1;
  // rs ;runbequtta no keisann kaisuu
 int rs = 15000;
 streamlines->setStreamParam(div, rs);
 streamlines->allocMem();
 //-----
 streamlines->loadData(xfile, yfile);
 double sp[3];
 float color[4];
 color[0]=color[1]=color[2]=color[3]=1.0f;
 sp[0] = 3.14; sp[1] = 1.2;
 for(int i=0;i<20;i++){
   sp[1] = 0.1+0.1*i;
   streamlines->stream_line(sp, color);
 }
 streamlines->writeBmp(picfile);
 //---
 streamlines->freeMem();
 
 delete streamlines;
 return 0;
}
