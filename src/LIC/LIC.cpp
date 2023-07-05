/* -- encoding UTF-8, LS -- */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <random>
// #include <bits/stdc++.h>

#include "lineintegralconv.h"

using namespace std;

const double PI = 3.1415926535897932;

double gridx;
double gridy;
const double lengx = 2.0 * PI;
const double lengy = 2.0;

int mag = 7;

LineIntegralConv *lic;

int main(int argc, char **argv) {
    if (argc <= 3)
        {  return 0;}
    char *xfile = argv[1];
    char *yfile = argv[2];
    char *picfile = argv[3];
    int gridx, gridy;
    if (argc == 6) {
        gridx = stoi(string(argv[4]));
        gridy = stoi(string(argv[5]));
    }
    else{
        gridx = 513;
        gridy = 1025;
    }
    double dx = lengx / (gridx - 1);
    double dy = lengy / (gridy - 1);

    lic = new LineIntegralConv;
    lic->setDebug(true);
    lic->setDatsize(gridx, gridy);
    lic->setDelta(dx, dy);
    lic->setwh(gridx, gridy, mag);

    double div = dy * 0.2;
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
