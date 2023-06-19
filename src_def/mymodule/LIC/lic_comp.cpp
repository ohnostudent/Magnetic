//
// vfive: Vector Field Interactive Visualization Environment
//
//
// lic_comp.c++
//                Computational Part of 2D Line Integral Convolution (LIC)
//                by Nobuaki Ohno  2005.10
//      

#include "./vect_visualizer.h"

extern Data *data;

////////////////////////////////////
// LIC (Line Integral Convolution)
//----------------------------------

////////////////////////////////////
// Initialization
//**********************************
void Lic::init()
{

    int i;

    //
    // Makeing white noise
    //

    white_noise = (float *) malloc(sizeof(float)
				   * SLICE_MAX_COORD * SLICE_MAX_COORD);

    for (i = 0; i < SLICE_MAX_COORD * SLICE_MAX_COORD; i++) {

	if (rand() > RAND_MAX / 2.0)
	    white_noise[i] = 0.99;
	else
	    white_noise[i] = 0.00;

//      white_noise[i] = rand() / (float) RAND_MAX;

    }


}

//
// free memory for noise
//
void Lic::end()
{
    free(white_noise);
}

//
// Main function
//
void Lic::makelicdata(int ax_n, int cut_n, int sv)
{
    int i, j;
    int x, y, z;
    int w_max, h_max;
    char filename[256];
    FILE *fp;
    float *out;
    int n1, n2, n3;

    n1 = data->n1;
    n2 = data->n2;
    n3 = data->n3;

    if (n1 > SLICE_MAX_COORD)
	n1 = SLICE_MAX_COORD;

    if (n2 > SLICE_MAX_COORD)
	n2 = SLICE_MAX_COORD;

    if (n3 > SLICE_MAX_COORD)
	n3 = SLICE_MAX_COORD;

    switch (ax_n) {
    case I:
	out = (float *) malloc(sizeof(float) * n2 * n3);
	bzero(out, sizeof(float) * n2 * n3);
	w_max = n2;
	h_max = n3;
	break;

    case J:
	out = (float *) malloc(sizeof(float) * n1 * n3);
	bzero(out, sizeof(float) * n1 * n3);
	w_max = n1;
	h_max = n3;
	break;

    case K:
	out = (float *) malloc(sizeof(float) * n1 * n2);
	bzero(out, sizeof(float) * n1 * n2);
	w_max = n1;
	h_max = n2;
	break;

    default:
	puts("Error");
	break;
    }


//    puts("LIC start !");

    omp_set_num_threads(LIC_NUM_THRDS);

#pragma omp parallel for schedule (dynamic, 2) private(x, y, z, i, j) shared(ax_n, h_max, w_max, cut_n, out)
    for (j = 1; j < h_max - 1; j++) {

	for (i = 1; i < w_max - 1; i++) {

	    switch (ax_n) {
	    case I:
		x = cut_n;
		y = i;
		z = j;
		break;

	    case J:
		x = i;
		y = cut_n;
		z = j;
		break;

	    case K:
		x = i;
		y = j;
		z = cut_n;
		break;

	    default:
		puts("Error");
		break;
	    }

	    lic_stream_lines(ax_n, cut_n, sv, x, y, z, out);

	}

//          if (j % 15 == 0)
//              printf("%d ends\n", j);
    }



    omp_set_num_threads(1);

    char filenameid[256];

    switch (ax_n) {
    case I:
	sprintf(filenameid, "%s0x", licfn);
	break;

    case J:
	sprintf(filenameid, "%s1x", licfn);
	break;

    case K:
	sprintf(filenameid, "%s2x", licfn);
	break;
    }


    //
    // Saving image data
    //
    sprintf(filename, "%s%st%d.dat", PolygonFilePath, filenameid,
	    data->ctime);

    if ((fp = fopen(filename, "wb")) == NULL)
	v5_error("File open error: lic.");

    fwrite(out, sizeof(float), w_max * h_max, fp);
    fclose(fp);

//    printf("%s\n", filename);
//    printf("Axis = %d, Cut_n = %d, width = %d, height = %d\n", ax_n, cut_n,
//         w_max, h_max);


    unsigned char pic_rgb[SLICE_MAX_COORD * SLICE_MAX_COORD * 3];

    for (i = 0; i < w_max * h_max; i++) {

	pic_rgb[3 * i + 0] = 255 * out[i];
	pic_rgb[3 * i + 1] = 255 * out[i];
	pic_rgb[3 * i + 2] = 255 * out[i];

    }


    fp = fopen("LIC_ppm.ppm", "w");

    if (fp == NULL)
	v5_error("File open error: image save.");

    fprintf(fp, "P6\n%d %d\n255\n", w_max, h_max);
    fwrite(pic_rgb, w_max * h_max * 3, 1, fp);

    fclose(fp);



    free(out);

    return;
}


///////////////////////////////
// LIC needs periodic filter
// Very Important. 
// If you want beautiful pics, 
// try to change this function
//-----------------------------
float Lic::lic_filter_function(float h, int i)
{
    float weight;
    float h_;

    if (h < 0.0)
	h_ = -h;
    else
	h_ = h;

    weight =
	cos(2.0 * PI * i * h_ / data->grid_size_min / 0.45) * exp(-i /
								  (double)
								  LIC_RUNGE_STEPS);

    if (weight < 0.0)
	weight = 0.0;

    return weight;
}

///////////////////////////////////////////////////
//
//
int Lic::lic_mesh_int_pos(float x, int *iout, int elemt){

    int n1, n2, n3;
    int o1, o2, o3;
    int lower, upper,middle;
    int found;


    switch (elemt) {

    case 0:
	n1 = data->n1;
	o1 = data->o1;

	if (x < data->x[o1] || x >= data->x[o1 + n1 - 1])
	    return 0;

	lower = o1 -1;
	upper = o1 + n1 - 1;
	middle = (lower + upper) / 2;

	found = 0;
	while (upper - lower > 1) {

	    if (data->x[middle] > x) {

		upper = middle;

	    } else if (data->x[middle + 1] <= x) {

		lower = middle;

	    } else {
		found = 1;
		*iout = middle;
		break;
	    }

	    middle = (lower + upper) / 2;

	}
	break;

	case 1:
	n2 = data->n2;
	o2 = data->o2;
	if (x < data->y[o2] || x >= data->y[n2 + o2 - 1])
	    return 0;

	lower = o2 -1;
	upper = o2 + n2 - 1;
	middle = (lower + upper) / 2;

	found = 0;
	while (upper - lower > 1) {

	    if (data->y[middle] > x) {

		upper = middle;

	    } else if (data->y[middle + 1] <= x) {

		lower = middle;

	    } else {
		found = 1;
		*iout = middle;
		break;
	    }

	    middle = (lower + upper) / 2;

	}
	break;

	case 2:
	n3 = data->n3;
	o3 = data->o3;
	if (x < data->z[o3] || x >= data->z[o3 + n3 - 1])
	    return 0;

	lower = o3 -1;
	upper = o3 + n3 - 1;
	middle = (lower + upper) / 2;

	found = 0;
	while (upper - lower > 1) {

	    if (data->z[middle] > x) {

		upper = middle;

	    } else if (data->z[middle + 1] <= x) {

		lower = middle;

	    } else {
		found = 1;
		*iout = middle;
		break;
	    }

	    middle = (lower + upper) / 2;

	}
	break;

     }


    if (found == 0){
	puts("Warning::can't find position");
	return 0;
	}

    return 1;

}

/*
static int comp( const void *target, const void *array ) {

    double *t = (double *)target;
    double *a = (double *)array;

    if( *t >= *( a + 1 )) {
	return 1;
    } else if( *t < *a ) {
	return -1;
    } else {
	return 0;
    }
}

int Lic::lic_mesh_int_pos(float x, int *iout, int elemt)
{

    int n1, n2, n3;
    int o1, o2, o3;

    double x_ = x;

    int size;
    double *ret;
    double *base;


    switch (elemt) {

    case 0:
	n1 = data->n1;
	o1 = data->o1;

	if (x_ < data->x[o1] || x_ >= data->x[o1 + n1 - 1])
	    return 0;

    size = data->n1 - 1;
    base = &data->x[o1];


    ret = (double *)bsearch( &x_, &data->x[o1],
		 size, sizeof(double), comp );

    if( ret == NULL ) {
	puts("*** error: mesh int pos\n");
	exit(9);
    } else {
	*iout = ret - base + o1;
    }


	break;

    case 1:
	n2 = data->n2;
	o2 = data->o2;
	if (x_ < data->y[o2] || x_ >= data->y[n2 + o2 - 1])
	    return 0;

    size = data->n2 - 1;
    base = &data->y[o2];

    ret = (double *)bsearch( &x_, &data->y[o2],
		 size, sizeof(double), comp );

    if( ret == NULL ) {
	puts("*** error: mesh int pos\n");
	exit(9);
    } else {
	*iout = ret - base + o2;
    }

	break;

    case 2:
	n3 = data->n3;
	o3 = data->o3;
	if (x_ < data->z[o3] || x_ >= data->z[o3 + n3 - 1])
	    return 0;

    size = data->n3 - 1;
    base = &data->z[o3];


    ret = (double *)bsearch( &x_, &data->z[o3],
		 size, sizeof(double), comp );

    if( ret == NULL ) {
	puts("*** error: mesh int pos\n");
	exit(9);
    } else {
	*iout = ret - base + o3;
    }


	break;

    }


    return 1;

}
*/

/*
int Lic::lic_mesh_int_pos(float x, int *iout, int elemt)
{

    int i, j, k;
    int n1, n2, n3;
    int o1, o2, o3;

    double half_x, half_y, half_z;
    double hhalf_x, hhalf_y, hhalf_z;

    int hn1, hn2, hn3;
    int hhn1, hhn2, hhn3;

    switch (elemt) {

    case 0:
	n1 = data->n1;
	o1 = data->o1;

	if (x < data->x[o1] || x >= data->x[o1 + n1 - 1])
	    return 0;

	hn1 = 0.75 * n1 + o1;
	hhn1 = 0.25 * n1 + o1;
	half_x = data->x[hn1];
	hhalf_x = data->x[hhn1];

	if (x >= half_x) {

	    for (i = n1 + o1 - 2; i >= o1; i--) {
		if (data->x[i] <= x) {
		    *iout = i;
		    break;
		}
	    }

	} else if (x >= hhalf_x) {

	    for (i = hn1 + 1; i >= o1; i--) {
		if (data->x[i] <= x) {
		    *iout = i;
		    break;
		}
	    }

	} else {

	    for (i = hhn1 + 1; i >= o1; i--) {
		if (data->x[i] <= x) {
		    *iout = i;
		    break;
		}
	    }


	}

	break;

    case 1:
	n2 = data->n2;
	o2 = data->o2;
	if (x < data->y[o2] || x >= data->y[n2 + o2 - 1])
	    return 0;

	hn2 = 0.75 * n2 + o2;
	hhn2 = 0.25 * n2 + o2;
	half_y = data->y[hn2];
	hhalf_y = data->y[hhn2];

	if (x >= half_y) {

	    for (j = n2 + o2 - 2; j >= o2; j--) {
		if (data->y[j] <= x) {
		    *iout = j;
		    break;
		}
	    }

	} else if (x >= hhalf_y) {

	    for (j = hn2 + 1; j >= o2; j--) {
		if (data->y[j] <= x) {
		    *iout = j;
		    break;
		}
	    }

	} else {

	    for (j = hhn2 + 1; j >= o2; j--) {
		if (data->y[j] <= x) {
		    *iout = j;
		    break;
		}
	    }

	}

	break;

    case 2:
	n3 = data->n3;
	o3 = data->o3;
	if (x < data->z[o3] || x >= data->z[o3 + n3 - 1])
	    return 0;

	hn3 = 0.75 * n3 + o3;
	hhn3 = 0.25 * n3 + o3;
	half_z = data->z[hn3];
	hhalf_z = data->z[hhn3];

	if (x >= half_z) {

	    for (k = n3 + o3 - 2; k >= o3; k--) {
		if (data->z[k] <= x) {
		    *iout = k;
		    break;
		}
	    }

	} else if (x >= hhalf_z) {

	    for (k = hn3 + 1; k >= o3; k--) {
		if (data->z[k] <= x) {
		    *iout = k;
		    break;
		}
	    }

	} else {

	    for (k = hhn3 + 1; k >= o3; k--) {
		if (data->z[k] <= x) {
		    *iout = k;
		    break;
		}
	    }
	}

	break;

    }


    return 1;

}
*/
/////////////////////////////////////////////
// get normed vector by linear interpolation
//-------------------------------------------
int Lic::lic_vect_value(int ax_n, int cut_n, float x, float y, float z,
			float *vecx, float *vecy, float *vecz, int sv)
{

    int intx, inty, intz;
    float dx, dy, dz;
    float one_minus_dx, one_minus_dy, one_minus_dz;
    float rv[8];
    int det;
    float absv;
    float vect_x, vect_y, vect_z;


    switch (ax_n) {
    case I:

	if (y < data->y[data->o2 + 1]
	    || y > data->y[data->o2 + data->n2 - 2]
	    || z < data->z[data->o3 + 1]
	    || z > data->z[data->o3 + data->n3 - 2])
	    return 0;

	x = data->x[cut_n];
	intx = cut_n;
	dx = 0.0;

	det = lic_mesh_int_pos(y, &inty, 1);
	if (det == 0)
	    return 0;
	dy = (y - data->y[inty]) / (data->y[inty + 1] - data->y[inty]);

	det = lic_mesh_int_pos(z, &intz, 2);
	if (det == 0)
	    return 0;
	dz = (z - data->z[intz]) / (data->z[intz + 1] - data->z[intz]);


	break;

    case J:
	if (x < data->x[data->o1 + 1]
	    || x > data->x[data->o1 + data->n1 - 2]
	    || z < data->z[data->o3 + 1]
	    || z > data->z[data->o3 + data->n3 - 2])
	    return 0;

	det = lic_mesh_int_pos(x, &intx, 0);
	if (det == 0)
	    return 0;
	dx = (x - data->x[intx]) / (data->x[intx + 1] - data->x[intx]);

	y = data->y[cut_n];
	inty = cut_n;
	dy = 0.0;

	det = lic_mesh_int_pos(z, &intz, 2);
	if (det == 0)
	    return 0;
	dz = (z - data->z[intz]) / (data->z[intz + 1] - data->z[intz]);
	break;

    case K:
	if (x < data->x[data->o1 + 1]
	    || x > data->x[data->o1 + data->n1 - 2]
	    || y < data->y[data->o2 + 1]
	    || y > data->y[data->o2 + data->n2 - 2])
	    return 0;

	det = lic_mesh_int_pos(x, &intx, 0);
	if (det == 0)
	    return 0;

	dx = (x - data->x[intx]) / (data->x[intx + 1] - data->x[intx]);

	det = lic_mesh_int_pos(y, &inty, 1);
	if (det == 0)
	    return 0;

	dy = (y - data->y[inty]) / (data->y[inty + 1] - data->y[inty]);

	z = data->z[cut_n];
	intz = cut_n;
	dz = 0.0;
	break;

    }

    one_minus_dx = 1.0 - dx;
    one_minus_dy = 1.0 - dy;
    one_minus_dz = 1.0 - dz;


    switch (ax_n) {
    case I:			/* Y-Z Plane */
	rv[0] = one_minus_dx * one_minus_dy * one_minus_dz;
	rv[2] = one_minus_dx * dy * one_minus_dz;
	rv[4] = one_minus_dx * one_minus_dy * dz;
	rv[6] = one_minus_dx * dy * dz;

	vect_x = 0.0;

	switch (data->vf) {
	case 0:
	    vect_y = rv[0] * data->vector_fld[sv]->y[intz][inty][intx]
		+ rv[2] * data->vector_fld[sv]->y[intz][inty + 1][intx]
		+ rv[4] * data->vector_fld[sv]->y[intz + 1][inty][intx]
		+ rv[6] * data->vector_fld[sv]->y[intz + 1][inty +
							    1][intx];

	    vect_z = rv[0] * data->vector_fld[sv]->z[intz][inty][intx]
		+ rv[2] * data->vector_fld[sv]->z[intz][inty + 1][intx]
		+ rv[4] * data->vector_fld[sv]->z[intz + 1][inty][intx]
		+ rv[6] * data->vector_fld[sv]->z[intz + 1][inty +
							    1][intx];
	    break;

	case 1:
	    vect_y = rv[0] * data->vector_fld[sv]->yf[intz][inty][intx]
		+ rv[2] * data->vector_fld[sv]->yf[intz][inty + 1][intx]
		+ rv[4] * data->vector_fld[sv]->yf[intz + 1][inty][intx]
		+ rv[6] * data->vector_fld[sv]->yf[intz + 1][inty +
							     1][intx];

	    vect_z = rv[0] * data->vector_fld[sv]->zf[intz][inty][intx]
		+ rv[2] * data->vector_fld[sv]->zf[intz][inty + 1][intx]
		+ rv[4] * data->vector_fld[sv]->zf[intz + 1][inty][intx]
		+ rv[6] * data->vector_fld[sv]->zf[intz + 1][inty +
							     1][intx];
	    break;
	}

	absv = sqrt((vect_y) * (vect_y) + (vect_z) * (vect_z));
	break;

    case J:			/* X-Z Plane */
	rv[0] = one_minus_dx * one_minus_dy * one_minus_dz;
	rv[1] = dx * one_minus_dy * one_minus_dz;
	rv[4] = one_minus_dx * one_minus_dy * dz;
	rv[5] = dx * one_minus_dy * dz;

	vect_y = 0.0;

	switch (data->vf) {
	case 0:
	    vect_x = rv[0] * data->vector_fld[sv]->x[intz][inty][intx]
		+ rv[1] * data->vector_fld[sv]->x[intz][inty][intx + 1]
		+ rv[4] * data->vector_fld[sv]->x[intz + 1][inty][intx]
		+ rv[5] * data->vector_fld[sv]->x[intz + 1][inty][intx +
								  1];

	    vect_z = rv[0] * data->vector_fld[sv]->z[intz][inty][intx]
		+ rv[1] * data->vector_fld[sv]->z[intz][inty][intx + 1]
		+ rv[4] * data->vector_fld[sv]->z[intz + 1][inty][intx]
		+ rv[5] * data->vector_fld[sv]->z[intz + 1][inty][intx +
								  1];
	    break;

	case 1:
	    vect_x = rv[0] * data->vector_fld[sv]->xf[intz][inty][intx]
		+ rv[1] * data->vector_fld[sv]->xf[intz][inty][intx + 1]
		+ rv[4] * data->vector_fld[sv]->xf[intz + 1][inty][intx]
		+ rv[5] * data->vector_fld[sv]->xf[intz + 1][inty][intx +
								   1];

	    vect_z = rv[0] * data->vector_fld[sv]->zf[intz][inty][intx]
		+ rv[1] * data->vector_fld[sv]->zf[intz][inty][intx + 1]
		+ rv[4] * data->vector_fld[sv]->zf[intz + 1][inty][intx]
		+ rv[5] * data->vector_fld[sv]->zf[intz + 1][inty][intx +
								   1];

	    break;

	}

	absv = sqrt((vect_x) * (vect_x) + (vect_z) * (vect_z));

	break;

    case K:			/* X-Y Plane */

	rv[0] = one_minus_dx * one_minus_dy * one_minus_dz;
	rv[1] = dx * one_minus_dy * one_minus_dz;

	rv[2] = one_minus_dx * dy * one_minus_dz;
	rv[3] = dx * dy * one_minus_dz;

	vect_z = 0.0;

	switch (data->vf) {
	case 0:
	    vect_x = rv[0] * data->vector_fld[sv]->x[intz][inty][intx]
		+ rv[1] * data->vector_fld[sv]->x[intz][inty][intx + 1]
		+ rv[2] * data->vector_fld[sv]->x[intz][inty + 1][intx]
		+ rv[3] * data->vector_fld[sv]->x[intz][inty + 1][intx +
								  1];

	    vect_y = rv[0] * data->vector_fld[sv]->y[intz][inty][intx]
		+ rv[1] * data->vector_fld[sv]->y[intz][inty][intx + 1]
		+ rv[2] * data->vector_fld[sv]->y[intz][inty + 1][intx]
		+ rv[3] * data->vector_fld[sv]->y[intz][inty + 1][intx +
								  1];
	    break;

	case 1:
	    vect_x = rv[0] * data->vector_fld[sv]->xf[intz][inty][intx]
		+ rv[1] * data->vector_fld[sv]->xf[intz][inty][intx + 1]
		+ rv[2] * data->vector_fld[sv]->xf[intz][inty + 1][intx]
		+ rv[3] * data->vector_fld[sv]->xf[intz][inty + 1][intx +
								   1];

	    vect_y = rv[0] * data->vector_fld[sv]->yf[intz][inty][intx]
		+ rv[1] * data->vector_fld[sv]->yf[intz][inty][intx + 1]
		+ rv[2] * data->vector_fld[sv]->yf[intz][inty + 1][intx]
		+ rv[3] * data->vector_fld[sv]->yf[intz][inty + 1][intx +
								   1];
	    break;
	}

	absv = sqrt((vect_x) * (vect_x) + (vect_y) * (vect_y));

	break;

    default:
	break;

    }


    if (absv >= 1.0E-20) {

	vect_x /= absv;
	vect_y /= absv;
	vect_z /= absv;

	*vecx = vect_x;
	*vecy = vect_y;
	*vecz = vect_z;

	return 1;

    } else {

	*vecx = 0.0;
	*vecy = 0.0;
	*vecz = 0.0;

	return 0;

    }

}


///////////////////////////////////////////
// determine color by linear-interpolation
//-----------------------------------------
int Lic::lic_set_colour(int ax_n, int cut_n, float *sx, float *intensity)
{

    int intx, inty, intz;
    float dx, dy, dz;
    float one_minus_dx, one_minus_dy, one_minus_dz;
    float rv[8];
    float value;
    int basic_point;
    float *color_pointer;
    float val1, val2, val3, val4;

    float x, y, z;

    x = sx[0];
    y = sx[1];
    z = sx[2];


    switch (ax_n) {
    case I:

	if (y < data->y[data->o2 + 1]
	    || y > data->y[data->o2 + data->n2 - 2]
	    || z < data->z[data->o3 + 1]
	    || z > data->z[data->o3 + data->n3 - 2])
	    return 0;

	intx = cut_n;
	dx = 0.0;

	inty =
	    (y - data->slice_coordy[0]) / (data->slice_coordy[2] -
					   data->slice_coordy[1]);

	dy = (y - data->slice_coordy[inty]) / (data->slice_coordy[2] -
					       data->slice_coordy[1]);

	intz =
	    (z - data->slice_coordz[0]) / (data->slice_coordz[2] -
					   data->slice_coordz[1]);

	dz = (z - data->slice_coordz[intz]) / (data->slice_coordz[2] -
					       data->slice_coordz[1]);

	break;

    case J:
	if (x < data->x[data->o1 + 1]
	    || x > data->x[data->o1 + data->n1 - 2]
	    || z < data->z[data->o3 + 1]
	    || z > data->z[data->o3 + data->n3 - 2])
	    return 0;

	inty = cut_n;
	dy = 0.0;

	intx =
	    (x - data->slice_coordx[0]) / (data->slice_coordx[2] -
					   data->slice_coordx[1]);

	dx = (x - data->slice_coordx[intx]) / (data->slice_coordx[2] -
					       data->slice_coordx[1]);

	intz =
	    (z - data->slice_coordz[0]) / (data->slice_coordz[2] -
					   data->slice_coordz[1]);

	dz = (z - data->slice_coordz[intz]) / (data->slice_coordz[2] -
					       data->slice_coordz[1]);

	break;

    case K:
	if (x < data->x[data->o1 + 1]
	    || x > data->x[data->o1 + data->n1 - 2]
	    || y < data->y[data->o2 + 1]
	    || y > data->y[data->o2 + data->n2 - 2])
	    return 0;

	intx =
	    (x - data->slice_coordx[0]) / (data->slice_coordx[2] -
					   data->slice_coordx[1]);

	dx = (x - data->slice_coordx[intx]) / (data->slice_coordx[2] -
					       data->slice_coordx[1]);

	inty =
	    (y - data->slice_coordy[0]) / (data->slice_coordy[2] -
					   data->slice_coordy[1]);

	dy = (y - data->slice_coordy[inty]) / (data->slice_coordy[2] -
					       data->slice_coordy[1]);

	intz = cut_n;
	dz = 0.0;
	break;

    }

    one_minus_dx = 1.0 - dx;
    one_minus_dy = 1.0 - dy;
    one_minus_dz = 1.0 - dz;

    switch (ax_n) {
    case I:			/* Y-Z Plane */
	basic_point = intz * SLICE_MAX_COORD + inty;
	rv[0] = one_minus_dy * one_minus_dz;
	rv[2] = dy * one_minus_dz;
	rv[4] = one_minus_dy * dz;
	rv[6] = dy * dz;

	color_pointer = white_noise + basic_point;
	val1 = *color_pointer;

	color_pointer += 1;
	val2 = *color_pointer;

	color_pointer += (SLICE_MAX_COORD - 1);
	val3 = *color_pointer;

	color_pointer += 1;
	val4 = *color_pointer;

	value = rv[0] * val1 + rv[2] * val2 + rv[4] * val3 + rv[6] * val4;

	break;

    case J:			/* X-Z Plane */
	basic_point = intz * SLICE_MAX_COORD + intx;
	rv[0] = one_minus_dx * one_minus_dz;
	rv[1] = dx * one_minus_dz;
	rv[4] = one_minus_dx * dz;
	rv[5] = dx * dz;

	color_pointer = white_noise + basic_point;
	val1 = *color_pointer;

	color_pointer += 1;
	val2 = *color_pointer;

	color_pointer += (SLICE_MAX_COORD - 1);
	val3 = *color_pointer;

	color_pointer += 1;
	val4 = *color_pointer;

	value = rv[0] * val1 + rv[1] * val2 + rv[4] * val3 + rv[5] * val4;

	break;

    case K:			/* X-Y Plane */
	basic_point = inty * SLICE_MAX_COORD + intx;
	rv[0] = one_minus_dx * one_minus_dy;
	rv[1] = dx * one_minus_dy;
	rv[2] = one_minus_dx * dy;
	rv[3] = dx * dy;

	color_pointer = white_noise + basic_point;
	val1 = *color_pointer;

	color_pointer += 1;
	val2 = *color_pointer;

	color_pointer += (SLICE_MAX_COORD - 1);
	val3 = *color_pointer;

	color_pointer += 1;
	val4 = *color_pointer;


	value = rv[0] * val1 + rv[1] * val2 + rv[2] * val3 + rv[3] * val4;

	break;

    }

    *intensity = value;

    return 1;
}

//
// Integration routine
//
void Lic::lic_stream_lines(int ax_n, int cut_n, int sv, int xx, int yy,
			   int zz, float *out)
{
    int i, direction;
    int if_continue;
    float h;
    float kx1, kx2, kx3, kx4;
    float ky1, ky2, ky3, ky4;
    float kz1, kz2, kz3, kz4;
    float sx[3];

    float vx, vy, vz;

    float sum_intensity, intensity;
    float weight, sum_weight;

    int pixel_pos, step_num;

    int n1 = data->n1;
    int n2 = data->n2;
    int n3 = data->n3;


    if (n1 > SLICE_MAX_COORD)
	n1 = SLICE_MAX_COORD;
    if (n2 > SLICE_MAX_COORD)
	n2 = SLICE_MAX_COORD;
    if (n3 > SLICE_MAX_COORD)
	n3 = SLICE_MAX_COORD;

    sum_intensity = 0.0;
    sum_weight = 0.0;

    switch (ax_n) {
    case I:
	pixel_pos = yy + zz * n2;
	break;

    case J:
	pixel_pos = xx + zz * n1;
	break;

    case K:
	pixel_pos = xx + yy * n1;
	break;
    }


// Runge Kutta 4
    for (direction = 0; direction < 2; direction++) {

	switch (ax_n) {
	case I:
	    sx[0] = data->x[xx];
	    sx[1] = data->slice_coordy[yy];
	    sx[2] = data->slice_coordz[zz];
	    break;

	case J:
	    sx[0] = data->slice_coordx[xx];
	    sx[1] = data->y[yy];
	    sx[2] = data->slice_coordz[zz];
	    break;

	case K:
	    sx[0] = data->slice_coordx[xx];
	    sx[1] = data->slice_coordy[yy];
	    sx[2] = data->z[zz];
	    break;
	}

//  printf("Mag_Line est %f\n",de_mag);

	switch (direction) {

	case 0:
	    h = data->grid_size_min * 0.2;
	    break;

	case 1:
	    h = -data->grid_size_min * 0.2;
	    break;

	}

	if (direction == 0) {

	    lic_set_colour(ax_n, cut_n, sx, &intensity);

	    weight = lic_filter_function(0.0, 0);

	    sum_intensity = intensity * weight;
	    sum_weight = weight;


	    vx = vy = vz = 0.0;
	    if_continue =
		lic_vect_value(ax_n, cut_n, sx[0], sx[1], sx[2], &vx, &vy,
			       &vz, sv);

	    i = 0;

	    if (if_continue == 0)
		break;
	}

	step_num = LIC_RUNGE_STEPS;

	for (i = 1; i < step_num; i++) {

	    //if(h<0.0) break;

	    if_continue =
		lic_vect_value(ax_n, cut_n, sx[0], sx[1], sx[2], &vx, &vy,
			       &vz, sv);

	    if (if_continue == 0)
		break;

	    kx1 = h * vx;
	    ky1 = h * vy;
	    kz1 = h * vz;


	    if_continue =
		lic_vect_value(ax_n, cut_n, sx[0] + 0.5 * kx1,
			       sx[1] + 0.5 * ky1, sx[2] + 0.5 * kz1, &vx,
			       &vy, &vz, sv);

	    if (if_continue == 0)
		break;

	    kx2 = h * vx;
	    ky2 = h * vy;
	    kz2 = h * vz;


	    if_continue =
		lic_vect_value(ax_n, cut_n, sx[0] + 0.5 * kx2,
			       sx[1] + 0.5 * ky2, sx[2] + 0.5 * kz2, &vx,
			       &vy, &vz, sv);

	    if (if_continue == 0)
		break;

	    kx3 = h * vx;
	    ky3 = h * vy;
	    kz3 = h * vz;


	    if_continue =
		lic_vect_value(ax_n, cut_n, sx[0] + kx3, sx[1] + ky3,
			       sx[2] + kz3, &vx, &vy, &vz, sv);

	    if (if_continue == 0)
		break;

	    kx4 = h * vx;
	    ky4 = h * vy;
	    kz4 = h * vz;


	    sx[0] += (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4) / 6.0;
	    sx[1] += (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4) / 6.0;
	    sx[2] += (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4) / 6.0;


	    lic_set_colour(ax_n, cut_n, sx, &intensity);

	    weight = lic_filter_function(h, i);

	    sum_intensity += (intensity * weight);
	    sum_weight += weight;

	}			/* i */


    }				/* direction */

    if (i != 0)
	out[pixel_pos] = sum_intensity / sum_weight;
    else
	out[pixel_pos] = 0.0;
}
