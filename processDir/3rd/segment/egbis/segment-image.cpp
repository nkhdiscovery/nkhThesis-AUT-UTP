/*
Copyright (C) 2015 Yasutomo Kawanishi
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#include "segment-image.h"
#include <cmath>


// random color
rgb random_rgb(){ 
  rgb c;
  double r;
  
  c.r = (uchar)random();
  c.g = (uchar)random();
  c.b = (uchar)random();

  return c;
}

// dissimilarity measure between pixels
static inline float diff(image<float> *l, image<float> *a, image<float> *b,
             int x1, int y1, int x2, int y2) {

//  return sqrt(square(imRef(r, x1, y1)-imRef(r, x2, y2)) +
//          square(imRef(g, x1, y1)-imRef(g, x2, y2)) +
//          square(imRef(b, x1, y1)-imRef(b, x2, y2)));


    //TODO: 94 and 2000 from https://en.wikipedia.org/wiki/Color_difference,
    float l1 = imRef(l, x1, y1), l2 = imRef(l, x2, y2),
              a1 = imRef(a, x1, y1), a2 = imRef(a, x2, y2),
              b1 = imRef(b, x1, y1), b2 = imRef(b, x2, y2);

    return sqrt(square(l1-l2)+50*square(b1-b2) );

//    return sqrt(square(b1-b2));

    //return sqrt(square(l1-l2) + square(a1-a2) + 100*square(b1-b2));// + square(b2-b1) + square(a1-a2)) ;

    //NOTE: White: late S (hsv), Early V (HSV) , Good S (HLS)
    // Bset test on 48: return sqrt(square(a1-a2) + square(b1-b2)); which is for hls
    // Best test on 48: return sqrt(square(b1-b2) ); which is for hsv and 1000 50
    // Even better with return sqrt(square(a1-a2) + square(square(b1-b2)) ); for both white and green. not good on 1 and 10, means still no white
    // So return sqrt(square(aq1-a2)); and
    //   also return sqrt(square(l1-l2) + square(a1-a2) + 100*square(b1-b2)); on hsv almost works for 10. with 200 & 100.
    // That's why I should consider shape
    // Try passing a multi-channel mixed from working channels above, like V from hsv and sth from LAB

    /*
    double C1 = sqrt(pow(a1, 2) + pow(b1, 2));
    double C2 = sqrt(pow(a2, 2) + pow(b2, 2));
    double H = pow(a1 - a2, 2) + pow(b1 - b2, 2) - pow(C1 - C2 , 2);
    return (sqrt(pow((l1 - l2) / 1 , 2)
                      + pow((C1- C2) / (1 + 0.45 * C1), 2) + H / pow ((1 + 0.15 * C1), 2)));
*/

  /*
  float l1 = imRef(l, x1, y1), l2 = imRef(l, x2, y2),
            a1 = imRef(a, x1, y1), a2 = imRef(a, x2, y2),
            b1 = imRef(b, x1, y1), b2 = imRef(b, x2, y2);
    double Pi2 = M_PI * 2;
    double L_ = (l1 + l2) / 2.0;
    double C1 = sqrt(pow(a1, 2) + pow(b1, 2));
    double C2 = sqrt(pow(a2, 2) + pow(b2, 2));
    double C_ = (C1 + C2) / 2.0;
    double G = (1 - sqrt((pow(C_ , 7)) / (pow(C_ , 7) + 6103515625))) / 2.0;
    double a1_ = a1 * (1 + G);
    double a2_ = a2 * (1 + G);
    double C1_ = sqrt(pow(a1_ , 2) + pow(b1, 2));
    double C2_ = sqrt(pow(a2_ , 2) + pow(b2, 2));
    double C__ = (C1_ + C2_) / 2.0;
    double h1_ = atan2(b1, a1_);
    h1_ = (h1_ < 0) ? h1_ + Pi2 : (h1_  >= Pi2) ? h1_ - Pi2 : h1_;
    double h2_ = atan2(b2, a2_);
    h2_ = (h2_ < 0) ? h2_ + Pi2 : (h2_  >= Pi2) ? h2_ - Pi2 : h2_;
    double H__ = (fabs(h1_ - h2_) > M_PI) ? (h1_ + h2_ + Pi2) / 2.0 : (h1_ + h2_) / 2.0;
    double T = 1 - 0.17 * cos(H__ - 0.5236) + 0.24 * cos(2 * H__) + 0.32 * cos(3 * H__ + 0.10472)  - 0.2 * cos(4 * H__ - 1.0995574);
    double dh_ = h2_ - h1_;
    dh_ = (abs(dh_) > M_PI && h2_  <= h1_) ? dh_ + Pi2 : (abs(dh_) > M_PI && h2_ > h1_) ? dh_ - Pi2 : dh_;
    double dH_ = 2 * sqrt(C1_ * C2_) * sin(dh_ / 2.0);
    double SL = 1 + ((0.015 * pow(L_ - 50, 2)) / (sqrt (20 + pow(L_ - 50, 2))));
    double SC = 1 + 0.045 * C__;
    double SH = 1 + 0.015 * C__ * T;
    double dO = 1.0471976 * exp(-pow(( H__ - 4.799655) /  0.436332313 ,  2));
    double RC = 2 * sqrt(pow(C__ , 7) / (pow(C__ , 7) + 6103515625));
    double RT = -RC * sin(2 * dO);
    return (sqrt(pow((l2 - l1) / SL , 2) + pow((C2_ - C1_) / SC , 2) + pow(dH_ / SH , 2) + RT * ((C2_ - C1_) / SC) * ((dH_) / SH)));
    */
}

/*
 * Segment an image
 *
 * Returns a color image representing the segmentation.
 *
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for treshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the segmentation.
 */
universe *segmentation(image<rgb> *im, float sigma, float c, int min_size,
			  int *num_ccs) {
  int width = im->width();
  int height = im->height();

  image<float> *r = new image<float>(width, height);
  image<float> *g = new image<float>(width, height);
  image<float> *b = new image<float>(width, height);

  // smooth each color channel
//#ifdef _OPENMP
//#pragma omp parallel for collapse(2)
//#endif
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      imRef(r, x, y) = imRef(im, x, y).r;
      imRef(g, x, y) = imRef(im, x, y).g;
      imRef(b, x, y) = imRef(im, x, y).b;
    }
  }
  image<float> *smooth_r = smooth(r, sigma);
  image<float> *smooth_g = smooth(g, sigma);
  image<float> *smooth_b = smooth(b, sigma);
  delete r;
  delete g;
  delete b;
 
  // build graph
  edge *edges = new edge[width*height*4];
  int num = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x < width-1) {
	edges[num].a = y * width + x;
	edges[num].b = y * width + (x+1);
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y);
	num++;
      }

      if (y < height-1) {
	edges[num].a = y * width + x;
	edges[num].b = (y+1) * width + x;
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x, y+1);
	num++;
      }

      if ((x < width-1) && (y < height-1)) {
	edges[num].a = y * width + x;
	edges[num].b = (y+1) * width + (x+1);
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y+1);
	num++;
      }

      if ((x < width-1) && (y > 0)) {
	edges[num].a = y * width + x;
	edges[num].b = (y-1) * width + (x+1);
	edges[num].w = diff(smooth_r, smooth_g, smooth_b, x, y, x+1, y-1);
	num++;
      }
    }
  }
  delete smooth_r;
  delete smooth_g;
  delete smooth_b;

  // segment
  universe *u = segment_graph(width*height, num, edges, c);
  
  // post process small components
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
  for (int i = 0; i < num; i++) {
    int a = u->find(edges[i].a);
    int b = u->find(edges[i].b);
    if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
      u->join(a, b);
  }
  delete [] edges;
  *num_ccs = u->num_sets();

  return u;
}

image<rgb>* visualize(universe *u, int width, int height){
  image<rgb> *output = new image<rgb>(width, height);

  // pick random colors for each component
  rgb *colors = new rgb[width*height];

  for (int i = 0; i < width*height; i++)
    colors[i] = random_rgb();
  
//#ifdef _OPENMP
//#pragma omp parallel for collapse(2)
//#endif
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int comp = u->find(y * width + x);
      imRef(output, x, y) = colors[comp];
    }
  }  

  delete [] colors;  
  return output;
}

image<rgb> *segment_image(image<rgb> *im, float sigma, float c, int min_size,
			  int *num_ccs) {
    universe *u = segmentation(im, sigma, c, min_size, num_ccs);
    image<rgb> *visualized = visualize(u, im->width(), im->height());
	delete u;
    return visualized;
}
