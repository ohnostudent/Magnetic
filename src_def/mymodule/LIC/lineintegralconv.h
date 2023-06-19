#ifndef _LIC_MODULE
#define _LIC_MODULE

using namespace std;

class LineIntegralConv {
    const double PI = 3.14159265358979;
    bool debug;
    int pic_w, pic_h, mag;
    int dat_size[2];
    double axis_min[2];
    double axis_delta[2];
    double mag_delta[2];
    double stream_div;
    int rungekutta_step;
    double region_min[3];
    double region_max[3];
    
    int *licpxl[2][2];
    int rnlicpxl[2];
    int numlicpxl;

    int find_fact(double x, double y, int *i, int *j, double cx[4], double cy[4]);
    double interpolator(int i, int j, double *cx, double *cy, int ele);
    int vec_value(double x, double y, double z, double *vx, double *vy, double *vz, double *vec_abs);

    //double c01, c02, c03, c04, c05, c06, c07, c08, c09, c10;
    //double c11, c12, c13, c14, c15, c16, c17, c18, c19, c20;
    //double c21, c22, c23, c24, c25, c26, c27;

    float *fx, *fy;
    float *rgb;
    float *whitenoise;

    float stream_line(int x0[2]);
    void pos2pxl(double *pos, int *pxl);
    void pxl2pos(int *pxl, double *pos);

  public:
    LineIntegralConv();
    void setDebug(bool);
    void setDatsize(int, int);
    void setDelta(double, double);
    void setAxisMin(double, double);
    void setwh(int, int, int);
    void setStreamParam(double, int);
    void allocMem();
    void freeMem();
    void preproc();
    void setLen(int);
    void loadData(char *, char *);
    void writeBmp(char *filename);
    void makeLIC();
};

#endif
