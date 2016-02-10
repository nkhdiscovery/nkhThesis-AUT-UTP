#ifndef NKHALGORITHMS_H
#define NKHALGORITHMS_H

#include <opencv/cv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
//#include <opencv2/ximgproc/segmentation.hpp>
#include <opencv2/core/utility.hpp>

#include <opencv2/bioinspired.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp>


//using namespace cv;
//using namespace cv::ximgproc::segmentation;

//Edge aware smoothings
#include "guidedfilter.h"

//3rd dtf
#include "tclap/CmdLine.h"
#include "Image.h"
#include "rdtsc.h"
#include "FunctionProfiling.h"
#include "NC.h"
#include "RF.h"


#include "FrameObjects.h"
#include "3rd/segment/egbis.h"



enum{
NORMAL_STATE,
INSUFFICIENT_ARGUMENTS,
NO_SUCH_FILE_OR_DIR,
NOT_A_DIR,
NOT_REGULAR_FILE,

};


/****************** nkhStart: macro utils ******************/
#define getStr(x) #x

#define checkPath(p) if(!exists(p)){\
    cerr << p << " : " << getStr(NO_SUCH_FILE_OR_DIR\n);\
    return NO_SUCH_FILE_OR_DIR; }
#define checkDir(p) checkPath(p);\
    if(!is_directory(p)){\
    cerr << p << " : " << getStr(NOT_A_DIR\n);\
    return NOT_A_DIR; }
#define checkRegularFile(p) checkPath(p);\
    if(!is_regular_file(p)){\
    cerr << p << " : " << getStr(NOT_REGULAR_FILE\n);\
    return NOT_REGULAR_FILE; }
/****************** nkhEnd: macro utils ******************/

/****************** nkhStart: global vars and defs ******************/
#define RESIZE_FACTOR 5
#define RESIZE_FACTOR2 5
#define _1080_x 1920
#define _1080_y 1080
#define _720_x 1280
#define _720_y 720


#define DOMAIN_SIGMA_S 100 //30 orig// 20 green & brown
#define DOMAIN_SIGMA_R 250 //350 orig //190 in green //290 brown
#define DOMAIN_MAX_ITER 3
#define DTF_METHOD "NC"


#define resizeAntRecFrom1080(rec) {rec.x /= RESIZE_FACTOR/1.5;\
    rec.y /= RESIZE_FACTOR/1.5;\
    rec.width /= RESIZE_FACTOR/1.5;\
    rec.height /= RESIZE_FACTOR/1.5;\
    } //1.5 cuz of 1080p to 720p annotation

#define resizeDetRecTo1080(rec) {rec.x *= RESIZE_FACTOR;\
    rec.y *= RESIZE_FACTOR;\
    rec.width *= RESIZE_FACTOR;\
    rec.height *= RESIZE_FACTOR;\
    } //1.5 cuz of 1080p to 720p annotation

/******************** nkhStart opencv ********************/
    void nkhImshow(const char* windowName, cv::Mat& img)
{
    imshow(windowName, img);
}

char maybeImshow(const char* windowName, cv::Mat& img, int waitKeyTime=20)
{
    if(WITH_VISUALIZATION)
    {
        nkhImshow(windowName, img);
        return cvWaitKey(waitKeyTime);
    }
    return 0;
}

cv::Scalar hsv_to_rgb(cv::Scalar c) {
    cv::Mat in(1, 1, CV_32FC3);
    cv::Mat out(1, 1, CV_32FC3);

    float * p = in.ptr<float>(0);

    p[0] = c[0] * 360;
    p[1] = c[1];
    p[2] = c[2];

    cv::cvtColor(in, out, cv::COLOR_HSV2RGB);

   cv::Scalar t;

    cv::Vec3f p2 = out.at<cv::Vec3f>(0, 0);

    t[0] = (int)(p2[0] * 255);
    t[1] = (int)(p2[1] * 255);
    t[2] = (int)(p2[2] * 255);

    return t;

}

cv::Scalar color_mapping(int segment_id) {

    double base = (double)(segment_id) * 0.618033988749895 + 0.24443434;

    return hsv_to_rgb(cv::Scalar(fmod(base, 1.2), 0.95, 0.80));

}


void mat2ToMat(Mat2<float3>& in, cv::Mat& out)
{
    out = cv::Mat(in.height, in.width, CV_32F, in.data);
}

void mserExtractor (const cv::Mat& image, cv::Mat& mserOutMask){
    cv::Ptr<cv::MSER> mserExtractor  = cv::MSER::create();

    vector<vector<cv::Point>> mserContours;
    vector<cv::Rect> mserBbox;
    mserExtractor->detectRegions(image, mserContours, mserBbox);

    for(unsigned int i = 0; i<mserContours.size(); i++ )
    {
        drawContours(mserOutMask, mserContours, i, cv::Scalar(255, 255, 255), 4);
    }
}

void on_trackbarS( int, void* )
{
}
void on_trackbarR( int, void* )
{

}
//int sigmaS=0, sigmaR=0;
void dtfWrapper(cv::Mat& in, cv::Mat& out)
{
    cv::ximgproc::dtFilter(in.clone(), in, out, DOMAIN_SIGMA_S, DOMAIN_SIGMA_R, cv::ximgproc::DTF_NC); //r 350. 50. nc
}

/*------------------- nkhEnd opencv -------------------*/

/******************** nkhStart domainTransform ********************/
// Recursive filter for vertical direction
void recursiveFilterVertical(cv::Mat& out, cv::Mat& dct, double sigma_H) {
    int width  = out.cols;
    int height = out.rows;
    int dim    = out.channels();
    double a   = exp(-sqrt(2.0) / sigma_H);

    cv::Mat V;
    dct.convertTo(V, CV_64FC1);
    for(int x=0; x<width; x++) {
        for(int y=0; y<height-1; y++) {
            V.at<double>(y, x) = pow(a, V.at<double>(y, x));
        }
    }

    // if openmp is available, compute in parallel
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int x=0; x<width; x++) {
        for(int y=1; y<height; y++) {
            double p = V.at<double>(y-1, x);
            for(int c=0; c<dim; c++) {
                double val1 = out.at<double>(y, x*dim+c);
                double val2 = out.at<double>(y-1, x*dim+c);
                out.at<double>(y, x*dim+c) = val1 + p * (val2 - val1);
            }
        }

        for(int y=height-2; y>=0; y--) {
            double p = V.at<double>(y, x);
            for(int c=0; c<dim; c++) {
                double val1 = out.at<double>(y, x*dim+c);
                double val2 = out.at<double>(y+1, x*dim+c);
                out.at<double>(y, x*dim+c) = val1 + p * (val2 - val1);
            }
        }
    }

    //Handle Memory
    V.release();
}

// Recursive filter for horizontal direction
void recursiveFilterHorizontal(cv::Mat& out, cv::Mat& dct, double sigma_H) {
    int width  = out.cols;
    int height = out.rows;
    int dim    = out.channels();
    double a = exp(-sqrt(2.0) / sigma_H);

    cv::Mat V;
    dct.convertTo(V, CV_64FC1);
    for(int x=0; x<width-1; x++) {
        for(int y=0; y<height; y++) {
            V.at<double>(y, x) = pow(a, V.at<double>(y, x));
        }
    }

    // if openmp is available, compute in parallel
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int y=0; y<height; y++) {
        for(int x=1; x<width; x++) {
            double p = V.at<double>(y, x-1);
            for(int c=0; c<dim; c++) {
                double val1 = out.at<double>(y, x*dim+c);
                double val2 = out.at<double>(y, (x-1)*dim+c);
                out.at<double>(y, x*dim+c) = val1 + p * (val2 - val1);
            }
        }

        for(int x=width-2; x>=0; x--) {
            double p = V.at<double>(y, x);
            for(int c=0; c<dim; c++) {
                double val1 = out.at<double>(y, x*dim+c);
                double val2 = out.at<double>(y, (x+1)*dim+c);
                out.at<double>(y, x*dim+c) = val1 + p * (val2 - val1);
            }
        }
    }

    //Handle Memory
    V.release();
}

// Domain transform filtering
void domainTransformFilter(cv::Mat& img, cv::Mat& out, cv::Mat& joint, double sigma_s, double sigma_r, int maxiter) {

    assert(img.depth() == CV_64F && joint.depth() == CV_64F);

    int width = img.cols;
    int height = img.rows;
    int dim = img.channels();

    // compute derivatives of transformed domain "dct"
    // and a = exp(-sqrt(2) / sigma_H) to the power of "dct"
    cv::Mat dctx = cv::Mat(height, width-1, CV_64FC1);
    cv::Mat dcty = cv::Mat(height-1, width, CV_64FC1);
    double ratio = sigma_s / sigma_r;

    for(int y=0; y<height; y++) {
        for(int x=0; x<width-1; x++) {
            double accum = 0.0;
            for(int c=0; c<dim; c++) {
                accum += abs(joint.at<double>(y, (x+1)*dim+c) - joint.at<double>(y, x*dim+c));
            }
            dctx.at<double>(y, x) = 1.0 + ratio * accum;
        }
    }

    for(int x=0; x<width; x++) {
        for(int y=0; y<height-1; y++) {
            double accum = 0.0;
            for(int c=0; c<dim; c++) {
                accum += abs(joint.at<double>(y+1, x*dim+c) - joint.at<double>(y, x*dim+c));
            }
            dcty.at<double>(y, x) = 1.0 + ratio * accum;
        }
    }

    // Apply recursive folter maxiter times
    img.convertTo(out, CV_MAKETYPE(CV_64F, dim));
    for(int i=0; i<maxiter; i++) {
        double sigma_H = sigma_s * sqrt(3.0) * pow(2.0, maxiter - i - 1) / sqrt(pow(4.0, maxiter) - 1.0);
        recursiveFilterHorizontal(out, dctx, sigma_H);
        recursiveFilterVertical(out, dcty, sigma_H);
    }

    //Handle Memory
    dctx.release();
    dcty.release();
}
void edgeAwareSmooth(cv::Mat& img, cv::Mat &dst);
/*------------------- nkhEnd domainTransform -------------------*/

/******************** nkhStart Saliency ********************/
void computeSaliency(cv::Mat& imgGray, int resizeFac, cv::Mat& saliencyMap)
{
    cv::Mat grayDown;
    std::vector<cv::Mat> mv;
    cv::Size resizedImageSize( resizeFac, resizeFac );

    cv::Mat realImage( resizedImageSize, CV_64F );
    cv::Mat imaginaryImage( resizedImageSize, CV_64F );
    imaginaryImage.setTo( 0 );
    cv::Mat combinedImage( resizedImageSize, CV_64FC2 );
    cv::Mat imageDFT;
    cv::Mat logAmplitude;
    cv::Mat angle( resizedImageSize, CV_64F );
    cv::Mat magnitude( resizedImageSize, CV_64F );
    cv::Mat logAmplitude_blur, imageGR;

    if( imgGray.channels() == 3 )
    {
        cv::cvtColor( imgGray, imageGR, cv::COLOR_BGR2GRAY );
        cv::resize( imageGR, grayDown, resizedImageSize, 0, 0, cv::INTER_LINEAR );
    }
    else
    {
        cv::resize( imgGray, grayDown, resizedImageSize, 0, 0, cv::INTER_LINEAR );
    }

    grayDown.convertTo( realImage, CV_64F );

    mv.push_back( realImage );
    mv.push_back( imaginaryImage );
    merge( mv, combinedImage );
    cv::dft( combinedImage, imageDFT );
    split( imageDFT, mv );

    //-- Get magnitude and phase of frequency spectrum --//
    cartToPolar( mv.at( 0 ), mv.at( 1 ), magnitude, angle, false );
    log( magnitude, logAmplitude );
    //-- Blur log amplitude with averaging filter --//
    cv::blur( logAmplitude, logAmplitude_blur, cv::Size( 3, 3 ), cv::Point( -1, -1 ), cv::BORDER_DEFAULT );

    exp( logAmplitude - logAmplitude_blur, magnitude );
    //-- Back to cartesian frequency domain --//
    polarToCart( magnitude, angle, mv.at( 0 ), mv.at( 1 ), false );
    merge( mv, imageDFT );
    cv::dft( imageDFT, combinedImage, cv::DFT_INVERSE );
    split( combinedImage, mv );

    cartToPolar( mv.at( 0 ), mv.at( 1 ), magnitude, angle, false );
    cv::GaussianBlur( magnitude, magnitude, cv::Size( 5, 5 ), 8, 0, cv::BORDER_DEFAULT );
    magnitude = magnitude.mul( magnitude );

    double minVal, maxVal;
    minMaxLoc( magnitude, &minVal, &maxVal );

    magnitude = magnitude / maxVal;
    magnitude.convertTo( magnitude, CV_32F );

    cv::resize( magnitude, saliencyMap, imgGray.size(), 0, 0, cv::INTER_LINEAR );

    // visualize saliency map
    //imshow( "Saliency Map Interna", saliencyMap );

    //Handle Memory
    grayDown.release();
    for (unsigned int i = 0 ; i < mv.size() ; i++)
        mv[i].release();
    mv.clear();
    realImage.release();
    imaginaryImage.release();
    combinedImage.release();
    imageDFT.release();
    logAmplitude.release();
    angle.release();
    magnitude.release();
    logAmplitude_blur.release();
    imageGR.release();
    //Free to go

}


void whiteThresh1(cv::Mat& edgeSmooth, cv::Mat& fin)
{
    //threshold
    cv::Mat chann[3], hls, hlsChann[3];
    cv::cvtColor(edgeSmooth, hls, CV_BGR2HLS);
    cv::split(hls, hlsChann);
    cv::split(edgeSmooth, chann);

    cv::Mat res1 =cv::Mat::zeros(chann[0].size().width, chann[0].size().height, CV_32F);
    cv::Mat res2=res1.clone(), res3=res1.clone(), tmp;

//    cv::Scalar mean0 = cv::mean(chann[0]), mean1 = cv::mean(chann[1]), mean2 = cv::mean(chann[2]);

    /*
    chann[0] -= mean0.val[0];
    chann[1] -= mean1.val[0];
    chann[2] -= mean2.val[0];
    */
//    cv::Mat reconst;

    cv::absdiff(chann[0], chann[1], res1);
    cv::absdiff(chann[1],chann[2], res2);
    cv::absdiff(chann[2],chann[0], res3);
    cv::add(res1, res2, tmp);
    cv::add(res3, tmp, tmp);

//    double minVal, maxVal;
//    cv::minMaxIdx(tmp, &minVal, &maxVal);
//        cout << minVal << ", " << maxVal << " Mean: " << cv::mean(tmp) << endl ;

//    cv::Mat newChan[3]={res1, res2, res3};
//    cv::merge( newChan, 3, reconst);
    cv::threshold(tmp, tmp, 65, 255, cv::THRESH_BINARY_INV);
    fin = tmp;
    fin = (hlsChann[1]>=100) & tmp;//worked
}

void whiteThresh2(cv::Mat& edgeSmooth, cv::Mat& saliencyOrig, cv::Mat& fin)
{
    cv::Mat saliency(saliencyOrig);
    saliency *= 120;
    //threshold
    cv::Mat chann[3], hls, hlsChann[3];
    cv::cvtColor(edgeSmooth, hls, CV_BGR2HLS);
    cv::split(hls, hlsChann);
    cv::split(edgeSmooth, chann);

    cv::Mat res1 =cv::Mat::zeros(chann[0].size().width, chann[0].size().height, CV_32F);
    cv::Mat res2=res1.clone(), res3=res1.clone(), tmp;

//    cv::Scalar mean0 = cv::mean(chann[0]), mean1 = cv::mean(chann[1]), mean2 = cv::mean(chann[2]);

    /*
    chann[0] -= mean0.val[0];
    chann[1] -= mean1.val[0];
    chann[2] -= mean2.val[0];
    */
    cv::Mat reconst;

    cv::absdiff(chann[0], chann[1], res1);
    cv::absdiff(chann[1],chann[2], res2);
    cv::absdiff(chann[2],chann[0], res3);
    cv::add(res1, res2, tmp);
    cv::add(res3, tmp, tmp);

    tmp.convertTo(tmp, CV_32F);
//    double minVal, maxVal;

//    cv::Mat newChan[3]={res1, res2, res3};
//    cv::merge( newChan, 3, reconst);
    cv::Mat tmpOrig(tmp);
    tmp /= 60.0; //D=20
//    cout << tmp << endl << "______________________________________" << endl;
//    tmpOrig = cv::Mat(tmp.size().height, tmp.size().width, CV_8U, cv::Scalar(255,255,255)) - tmpOrig;
//        cv::threshold(tmp, tmp, 20, 255, cv::THRESH_BINARY_INV);
        fin = (tmp) <1 ;//& (hlsChann[1]>=70) ;
//        cv::addWeighted(saliency, 0.5, fin, 0.5, -10, fin); //Play With it! //TODO
        cv::threshold(fin, fin, 170, 255, cv::THRESH_BINARY);

        return;

////        fin = (hlsChann[1]>=100) & tmp;//worked
//        return;
//    hlsChann[1].copyTo(tmp, hlsChann[1]>80);
//    cv::addWeighted(tmp, 0.7 , tmpOrig, 0.3, -10.0, fin);
//    cv::addWeighted(saliency, 0.8, fin, 0.7, -10, fin); //Play With it! //TODO
//    cv::threshold(fin, fin, 170, 255, cv::THRESH_BINARY);

}

void getMinS(cv::Mat& hls, cv::Mat* hlsChann, cv::Mat& hostMins)
{
    cv::cuda::GpuMat gpuHLS(hls), gpuHLSChann[3], gpuMinS, gpuTmp1(hlsChann[1]), gpuTmp2;
    cv::cuda::split(gpuHLS, gpuHLSChann);
    //0.01078941006 x^2 - 1.246292714 x + 57.70571406
    // 0.01649159736
    //6.467293081·10-3 x2 - 1.037263871 x + 130.4910273
    //2.038547409·10-5 x4 - 6.176184996·10-3 x3 + 6.480170616·10-1 x2 - 27.36243368 x + 470.4070268
    //Residual Sum of Squares: rss = 0

    //FINAL 7.135881581·10-3 x2 - 1.199965519 x + 139.9490098
    cv::cuda::multiply(gpuHLSChann[1], gpuHLSChann[1], gpuTmp2, 0.007135881581);
    gpuTmp1.convertTo(gpuTmp1, gpuTmp1.type(), 1.199965519);
    cv::cuda::subtract(gpuTmp2, gpuTmp1, gpuMinS);
    gpuTmp1 = cv::cuda::GpuMat(gpuTmp1.size(), gpuTmp1.type(), cv::Scalar(70));//139.9490098
    cv::cuda::add(gpuMinS, gpuTmp1, gpuMinS);
    gpuMinS.download(hostMins);

    cv::compare(hlsChann[2], hostMins, hostMins, cv::CMP_GE);
    hostMins = (hlsChann[2]>= 76) & (hlsChann[1]>=25) & hostMins;
}

void greenThresh1(cv::Mat& orig , cv::Mat& fin)
{
    cv::Mat hls;//, hlsChann[3];
    cv::cvtColor(orig, hls, CV_BGR2HLS);
    //cv::split(hls, hlsChann);
    cv::Mat tmp;
    cv::inRange(hls, cv::Scalar(75, 25, 63), cv::Scalar(85, 195, 255), tmp); //Threshold the color
    cv::threshold(tmp, tmp, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    fin = tmp.clone();
    return;
}


void brownThresh1(cv::Mat& orig , cv::Mat& fin)
{
    cv::Mat hls, hlsChann[3], bgrChan[3];
    cv::split(orig, bgrChan);
    cv::Mat bgDiff, tmp1;
    cv::absdiff(bgrChan[0], bgrChan[1], bgDiff);
    bgDiff = (bgDiff <=4);
    cv::addWeighted(bgrChan[0], 0.5, bgrChan[1], 0.5, 0, tmp1);
    cv::subtract(tmp1, bgrChan[2], tmp1, bgDiff);
    tmp1 = (tmp1<=10) & (tmp1>=4);

    cv::cvtColor(orig, hls, CV_BGR2HLS);
    cv::split(hls, hlsChann);
    cv::inRange(hls, cv::Scalar(7, 20, 63), cv::Scalar(14.5, 192, 224), fin); //Threshold the color
    //cv::inRange(hls, cv::Scalar(12.5, 20, 63), cv::Scalar(14.5, 92, 204), fin); //Threshold the color   LESS REGS
//    fin = (tmp1 & (hlsChann[1]<155) & (hlsChann[2]< 100) & (hlsChann[1]>=25)) | fin ;//& (hlsChann[0]<=90);// (hlsChann[2]>=15)& ;
//    fin |= (hlsChann[1]<43) & (hlsChrann[2]<128) & (hlsChann[0] <89);
    return;
}

void getDFTMag(cv::Mat& I, cv::Mat& complexI, cv::Mat& magI)
{
    cv::Mat padded(I);                            //expand input image to optimal size
//    int m = cv::getOptimalDFTSize( I.rows );
//    int n = cv::getOptimalDFTSize( I.cols ); // on the border add zero values
//    cv::copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    cv::dft(complexI, complexI);            // this way the result may fit in the source matrix


    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    cv::split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

    cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    magI = planes[0];

    magI += cv::Scalar::all(1);                    // switch to logarithmic scale
    cv::log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    cv::normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).

}

void getDFTInv(cv::Mat& complexI, cv::Mat& inverseTransform)
{
    //calculating the idft
    cv::dft(complexI, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
    //imshow("Reconstructed", inverseTransform);
}

//CUDA FFT is lower than CPU with Intel(R) Core(TM) i7-4702MQ CPU @ 2.20GHz, opencv with TBB. almost 3 times slower
void getDFTMag_CUDA(cv::Mat& hostImg, cv::Mat& hostMagI)
{
    cv::cuda::GpuMat I(hostImg), magI;
    cv::cuda::GpuMat padded;                            //expand input image to optimal size
    int m = cv::getOptimalDFTSize( I.rows );
    int n = cv::getOptimalDFTSize( I.cols ); // on the border add zero values
    cv::cuda::copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::cuda::GpuMat planes[] = {cv::cuda::GpuMat(padded), cv::cuda::GpuMat(padded)};

    planes[0].convertTo(planes[0], CV_32FC1);
    planes[1].convertTo(planes[1], CV_32FC1);
    cv::cuda::GpuMat complexI;
    cv::cuda::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    cv::cuda::dft(complexI, complexI, padded.size());            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    cv::cuda::split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::cuda::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    magI = planes[0];

    cv::cuda::add(magI, cv::Scalar::all(1), magI);                    // switch to logarithmic scale
    cv::cuda::log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    cv::cuda::GpuMat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::cuda::GpuMat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::cuda::GpuMat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::cuda::GpuMat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::cuda::GpuMat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    cv::cuda::normalize(magI, magI, 0, 1, CV_MINMAX, -1); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
    magI.download(hostMagI);
}


cv::Mat lookupTable(int levels) {
    int factor = 256 / levels;
    cv::Mat table(1, 256, CV_8U);
    uchar *p = table.data;

    for(int i = 0; i < 128; ++i) {
        p[i] = factor * (i / factor);
    }

    for(int i = 128; i < 256; ++i) {
        p[i] = factor * (1 + (i / factor)) - 1;
    }

    return table;
}
cv::Mat colorReduce(const cv::Mat &image, int levels) {
    cv::Mat table = lookupTable(levels);

    std::vector<cv::Mat> c;
    cv::split(image, c);
    for (std::vector<cv::Mat>::iterator i = c.begin(), n = c.end(); i != n; ++i) {
        cv::Mat &channel = *i;
        cv::LUT(channel.clone(), table, channel);
    }

    cv::Mat reduced;
    cv::merge(c, reduced);
    return reduced;
}

void nkhKmeans(cv::Mat& img, cv::Mat& dst)
{
    dst=img.clone();
    int channels= dst.channels();
    int pointsNum = dst.rows * dst.cols ;
    cv::Mat points(dst.total(), 3, CV_32F);

    points = dst.reshape(channels,pointsNum).clone() ;
    points.convertTo(points,CV_32F);
    cv::Mat labels;
    cv::Mat centers;

    cv::kmeans(points, 16, labels,
               cv::TermCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER,
                                300, 1), 1,cv::KMEANS_PP_CENTERS, centers);

    // map the centers
    cv::Mat new_image(dst.size(), dst.type());
    for( int row = 0; row != dst.rows; ++row){
        auto new_image_begin = new_image.ptr<uchar>(row);
        auto new_image_end = new_image_begin + new_image.cols * 3;
        auto labels_ptr = labels.ptr<int>(row * dst.cols);

        while(new_image_begin != new_image_end){
            int const cluster_idx = *labels_ptr;
            auto centers_ptr = centers.ptr<float>(cluster_idx);
            new_image_begin[0] = centers_ptr[0];
            new_image_begin[1] = centers_ptr[1];
            new_image_begin[2] = centers_ptr[2];
            new_image_begin += 3; ++labels_ptr;
        }
    }
    dst=new_image.clone();
}

void drawOptFlowMap (const cv::Mat& flow, cv::Mat& cflowmap, int step, const cv::Scalar& color) {
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const cv::Point2f& fxy = flow.at< cv::Point2f>(y, x);
            line(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            cv::circle(cflowmap, cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 1, color, -1);
        }
}

/*------------------- nkhEnd Saliency -------------------*/
void evaluateNonMasked(cv::Mat& binMask, map<int, FrameObjects>& groundTruth, int frameNum, vector<double>& result); //binMask should be binary
void evaluateMasked(cv::Mat& masked, map<int, FrameObjects>& groundTruth, int frameNum, vector<double>& result);
void calcMeanVar(vector<double>& result, string name="", bool printLn=true)
{
    double sum = std::accumulate(result.begin(), result.end(), 0.0);
    double mean = sum / result.size();

    double sq_sum = std::inner_product(result.begin(), result.end(), result.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / result.size() - mean * mean);
    cout << name << " ," << mean << "," << stdev ;
    if(printLn)
        cout << endl;
    else
        cout<< ",";
}

#endif // NKHALGORITHMS_H
