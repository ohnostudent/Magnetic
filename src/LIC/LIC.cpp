#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

#include "lineintegralconv.h"

using namespace std;

const double PI = 3.1415926535897932;

const double gridx = 513;
const double gridy = 1025;
const double lengx = 2.0*PI;
const double lengy = 2.0;

const double dx = lengx/(gridx-1);
const double dy = lengy/(gridy-1);

int mag = 7;

LineIntegralConv *lic;

int main(int argc, char **argv){
 char xfile[] = "magfield1";
 char yfile[] = "magfield2";
 char picfile[] = "lictest.bmp";
 
 lic = new LineIntegralConv;
 lic->setDebug(true);
 lic->setDatsize(gridx, gridy);
 lic->setDelta(dx, dy);
 lic->setwh(gridx, gridy, mag);
 double div = dy*0.2;
 int rs = 15000;
 lic->setStreamParam(div, rs);
 int licn = 100;
 lic->setLen(licn);
 lic->allocMem();
 lic->loadData(xfile, yfile);
 lic->preproc();
 lic->makeLIC();
 lic->writeBmp(picfile);
 lic->freeMem();
 
 delete lic;
    
 return 0;
}
