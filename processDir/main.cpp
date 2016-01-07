#include <QCoreApplication>
#include <opencv/cv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/bioinspired.hpp>
#include <opencv2/contrib/retina.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaobjdetect.hpp>

//using namespace cv;
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
using namespace boost::accumulators;
#include <fstream>
#include <map>
//using namespace std;

#include "nkhUtil.h"
#include "FrameObjects.h"

//Edge aware smoothings
#include "guidedfilter.h"

//3rd dtf
#include "tclap/CmdLine.h"
#include "Image.h"
#include "rdtsc.h"
#include "FunctionProfiling.h"
#include "NC.h"
#include "RF.h"

/****************** nkhStart: global vars and defs ******************/
#define RESIZE_FACTOR 7
#define _1080_x 1920
#define _1080_y 1080
#define _720_x 1280
#define _720_y 720

#define resizeAntRecFrom1080(rec) {rec.x /= RESIZE_FACTOR/1.5;\
    rec.y /= RESIZE_FACTOR/1.5;\
    rec.width /= RESIZE_FACTOR/1.5;\
    rec.height /= RESIZE_FACTOR/1.5;\
    } //1.5 cuz of 1080p to 720p annotation
#define DOMAIN_SIGMA_S 30 //30 orig// 20 green & brown
#define DOMAIN_SIGMA_R 270 //350 orig //190 in green //290 brown
#define DOMAIN_MAX_ITER 3
#define DTF_METHOD "RC"

/*----------------- nkhEnd: global vars and defs -----------------*/

void nkhMain(path inVid, path inFile, path outDir);

/****************** nkhStart: FPS counter ******************/
//TODO: implement as a template
time_t timerStart, timerEnd;
int timerCounter = 0;
double timerSec;
double timerFPS;
void initFPSTimer(){
    timerCounter = 0;
}
void fpsCalcStart()
{
    if (timerCounter == 0){
        time(&timerStart);
    }
}
string fpsCalcEnd()
{
    char* fpsString = new char[32];
    time(&timerEnd);
    timerCounter++;
    timerSec = difftime(timerEnd, timerStart);
    timerFPS = timerCounter/timerSec;
    if (timerCounter > 30)
        sprintf(fpsString, "%.2f fps\n",timerFPS);
    // overflow protection
    if (timerCounter == (INT_MAX - 1000))
        timerCounter = 0;
    return string(fpsString);
}
/*----------------- nkhEnd: FPS counter -----------------*/

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

void matToMat2(cv::Mat& in, Mat2<float3>& out){
    //out.data = in.data;
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

    cv::absdiff(chann[0], chann[1], res1);
    cv::absdiff(chann[1],chann[2], res2);
    cv::absdiff(chann[2],chann[0], res3);
    cv::add(res1, res2, tmp);
    cv::add(res3, tmp, tmp);

    cv::Mat tmpOrig(tmp);

    tmpOrig = cv::Mat(tmp.size().height, tmp.size().width, CV_8U, cv::Scalar(255,255,255)) ;//- tmpOrig;
    /*
//        fin = tmp;
//        fin = (hlsChann[1]>=100) & tmp;//worked
*/
    hlsChann[1].copyTo(tmpOrig, hlsChann[1]<=87);
    cv::threshold(tmpOrig, tmp, 1, 255, cv::THRESH_BINARY);
    tmp &= (hlsChann[2]<=29);
    tmp |= (hlsChann[2]<=36 & hlsChann[1]>=177);
//    cv::addWeighted(tmp, 0.7 , tmpOrig, 0.3, -10.0, fin);
    //cv::addWeighted(saliency, 0.8, fin, 0.7, -10, fin); //Play With it! //TODO
//    cv::threshold(fin, fin, 170, 255, cv::THRESH_BINARY);
    fin = tmp.clone();
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
    cv::inRange(hls, cv::Scalar(75, 25, 63), cv::Scalar(92, 195, 255), tmp); //Threshold the color
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
    cv::inRange(hls, cv::Scalar(0, 25, 66), cv::Scalar(21, 216, 255), fin); //Threshold the color
    fin = tmp1 & (hlsChann[1]<155) & (hlsChann[2]< 100) & (hlsChann[1]>=25) | fin ;//& (hlsChann[0]<=90);// (hlsChann[2]>=15)& ;
//    fin |= (hlsChann[1]<43) & (hlsChann[2]<128) & (hlsChann[0] <89);
    return;
}

void connectedMask(cv::Mat& smoothed, cv::Mat& conMap)
{
//    cv::cuda::GpuMat devSmooth(smoothed), devMap;
//    cv::cuda::connectivityMask(devSmooth, devMap, cv::Scalar::all(0), cv::Scalar::all(2));
//    devMap.download(conMap);
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

/*------------------- nkhEnd Saliency -------------------*/

void evaluateMasked(cv::Mat& masked, map<int, FrameObjects>& groundTruth, int frameNum, vector<double>& result);
void calcMeanVar(vector<double>& result)
{
    double sum = std::accumulate(result.begin(), result.end(), 0.0);
    double mean = sum / result.size();

    double sq_sum = std::inner_product(result.begin(), result.end(), result.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / result.size() - mean * mean);
    cout << "Mean is " << mean << ", stdev is: " << stdev << endl;
}

//void nkhTest();
void cropGroundTruth(cv::VideoCapture cap, path inFile, path outDir);

void parseFile(path inFile, map<int, FrameObjects>& outMap);

int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);
    
    if (WITH_CUDA == 1) {
        cerr << "USING CUDA\n";
//        cv::gpu::setDevice(0);
    } else {
        cerr << "NOT USING CUDA\n";
    }
    
    if (argc == 4)
    {
        path inVid(argv[1]), inFile(argv[2]), outDir(argv[3]);
        checkRegularFile(inVid);
        checkRegularFile(inFile);
        checkDir(outDir);
        nkhMain(inVid, inFile, outDir);
        
        return NORMAL_STATE;
    }
    else
    {
        cerr << getStr(error: INSUFFICIENT_ARGUMENTS\n);
        return INSUFFICIENT_ARGUMENTS;
    }
    //return a.exec();
}


void nkhMain(path inVid, path inFile, path outDir)
{
    //Open the video file
    cv::VideoCapture cap(inVid.string());
    
    //TODO: parse commandline arguments for different tasks
    //task1 : cropGroundTruth(VideoCapture cap, path inFile, path outDir)
    //task1 is done, ran once.
    
    //Task 1.5: open the map for evaluation
    map<int, FrameObjects> groundTruth;
    parseFile(inFile, groundTruth);

    //task2: resize and preproc.
    cv::Mat currentFrame;
    int frameCount = 0;
    initFPSTimer();
    vector<double> evalSaliency;


    /*

    cv::namedWindow("Smoothing S", 1);
    cv::namedWindow("Smoothing R", 1);

    //DT trackbars
    char TrackbarNameS[50], TrackbarNameR[50];
    sprintf( TrackbarNameS, "SigmaS: %d", sigmaS);
    sprintf( TrackbarNameR, "SigmaR: %d", sigmaR);
    cv::createTrackbar( TrackbarNameS, "Smoothing S", &sigmaS, 1500, on_trackbarS );
    cv::createTrackbar( TrackbarNameR, "Smoothing R", &sigmaR, 1500, on_trackbarR );

    /// Show some stuff
    on_trackbarS( sigmaS, 0 );
    on_trackbarR( sigmaR, 0 );
    */

/*
    cv::Mat template1 = cv::imread("1.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat tmplGray;
    cv::resize(template1, template1, cv::Size(template1.size().width/1, template1.size().height/1));
    cv::cvtColor(template1, tmplGray, cv::COLOR_BGR2GRAY);
    cv::threshold(tmplGray, tmplGray, 200, 255, cv::THRESH_BINARY);

    if(template1.empty())
    {
        cerr << "tmpl err\n";
        return;
    }
    */


    while(true)
    {
        cap >> currentFrame;
        if(currentFrame.size().area() <= 0)
        {
            cap.set(CV_CAP_PROP_POS_AVI_RATIO , 0);
            continue;
        }
        fpsCalcStart();
        
        cv::Mat frameResized, frameResized_gray, frameResizedHalf;
        cv::resize(currentFrame, frameResized, cv::Size(currentFrame.size().width/RESIZE_FACTOR,
                                             currentFrame.size().height/RESIZE_FACTOR));
        cv::resize(currentFrame, frameResizedHalf, cv::Size(currentFrame.size().width/2,
                                             currentFrame.size().height/2));
        cv::cvtColor(frameResized, frameResized_gray, cv::COLOR_BGR2GRAY);

        //TODO: implement with GPU
        //Mat edgeSmooth = measure<std::chrono::milliseconds>(edgeAwareSmooth, frameResized);
        cv::Mat edgeSmooth;
        dtfWrapper(frameResized, edgeSmooth);

        cv::Mat edgeSmoothLow;
        cv::ximgproc::dtFilter(edgeSmooth, frameResized, edgeSmoothLow, 10, 10, cv::ximgproc::DTF_NC); //r 350. 50. nc

        edgeSmooth = colorReduce(edgeSmooth, 64);
        edgeSmoothLow = colorReduce(edgeSmoothLow, 64);

        //SaliencyMap
        cv::Mat saliency, saliency72;
        computeSaliency(frameResized, 55 , saliency);
        computeSaliency(frameResized, 64 , saliency72);
        cv::Mat masked, binMask;
        saliency = saliency * 3;
        saliency72 = saliency72 * 3;
        saliency.convertTo( saliency, CV_8U );
        saliency72.convertTo( saliency72, CV_8U );
        // adaptative thresholding using Otsu's method, to make saliency map binary
        threshold( saliency, binMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU );
        //edgeAwareSmooth(frameResized, edgeSmooth);


        //threshold
        cv::Mat fin, whiteMask;

        whiteThresh2(edgeSmooth, saliency72, whiteMask);

        cv::Mat greenMask;
        greenThresh1(edgeSmoothLow, greenMask);

        cv::Mat brownMask;
        brownThresh1(edgeSmoothLow, brownMask);

        /*
        //TODO: Weight if needed

//        cv::Mat fin2;
//        greenThresh1(edgeSmoothLow, edgeSmooth, saliency, fin2);
//        cv::addWeighted(fin, 0.2, fin2, 0.2, 0 , fin);
//        cv::threshold(fin, greenMask, 100, 255, cv::THRESH_BINARY);

        //TODO: eval
        //evaluate Correlation
        //evaluateMasked(binMask, groundTruth, frameCount, evalSaliency);

        */


        int erosionDilation_size = 3;
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                                    cv::Size(2*erosionDilation_size + 1,
                                                             2*erosionDilation_size+1));

        cv::Mat colorMask = whiteMask ;//| greenMask ;//| brownMask ;

        /*
        cv::blur(colorMask, colorMask, cv::Size(12,12));
        cv::erode(colorMask, colorMask, element);
        cv::dilate(colorMask, colorMask, element);
        */
        edgeSmoothLow.copyTo(masked, colorMask);
        cv::Mat twoThird = cv::Mat::zeros(edgeSmooth.rows, edgeSmooth.cols, edgeSmooth.type());
        twoThird(cv::Rect(0,0, twoThird.size().width, 2*twoThird.size().height/3.0)) = cv::Scalar::all(255);
        masked.copyTo(twoThird, twoThird);
        computeSaliency(twoThird, 32, saliency);
        saliency *= 3;
        saliency.convertTo( saliency, CV_8U );
        threshold( saliency, binMask, 0, 255, cv::THRESH_BINARY );
        fin = colorMask ;//& binMask;
        masked.release();
        twoThird.copyTo(masked, fin);

        /*
         * GPU
         * int erosionDilation_size = 5;
    Mat element = cv::getStructuringElement(MORPH_RECT, Size(2*erosionDilation_size + 1,    2*erosionDilation_size+1));

    cuda::GpuMat d_element(element);
    cuda::GpuMat d_img(img);

    Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, d_img.type(), element);
    dilateFilter->apply(d_img, d_img);


         */


        //MSER
/*
        std::vector< std::vector< cv::Point> > contours;
        std::vector< cv::Rect> bboxes;
        // Ptr< MSER> mser = MSER::create(21, (int)(0.00002*textImg.cols*textImg.rows),
        //(int)(0.05*textImg.cols*textImg.rows), 1, 0.7);
        cv::Ptr< cv::MSER> mser = cv::MSER::create(100);
        mser->detectRegions(masked, contours, bboxes);
        for (int i = 0; i < bboxes.size(); i++)
        {
            cv::rectangle(masked, bboxes[i], CV_RGB(0, 255, 0));
        }
*/

        //contour appx

        /*
        cv::Mat edgeImg;
        cv::cuda::GpuMat devSmooth(edgeSmoothLow), devEqHist, devGraySmooth, devEdgeImg;
        cv::cuda::cvtColor(devSmooth, devGraySmooth, cv::COLOR_BGR2GRAY);
        cv::cuda::equalizeHist(devGraySmooth, devEqHist);
        cv::Ptr<cv::cuda::CannyEdgeDetector> cudaCanny = cv::cuda::createCannyEdgeDetector(50, 150);
        cudaCanny->detect(devEqHist, devEdgeImg);
        devEdgeImg.download(edgeImg);
        */

        //Visualize


        char controlChar = maybeImshow("Final", edgeSmoothLow) ;
        controlChar = maybeImshow("Kmeans", masked );

        if (controlChar == 'q')
        {
            break;
        }
        else if (controlChar == 'p')
        {
            while (cvWaitKey(10) != 'p');
        }
        else if(controlChar == 'r')
        {
            cap.set(CV_CAP_PROP_POS_AVI_RATIO , 0);
        }

        fpsCalcEnd();
        cout<< timerFPS << endl;
        frameCount++;

        //Handle Memory
//        frameResized.release();
//        frameResized_gray.release();
//        saliency.release();
//        masked.release();
//        binMask.release();
//        edgeSmooth.release();
        //Free to go

    }
    calcMeanVar(evalSaliency);

    //Handle Memory
    cap.release();
    currentFrame.release();
    groundTruth.clear();
    evalSaliency.clear();
    //Free to Go!

    return;
}

void evaluateMasked(cv::Mat& binMask, map<int, FrameObjects>& groundTruth, int frameNum, vector<double>& result) //binMask should be binary
{
    map<int, FrameObjects>::iterator it = groundTruth.find(frameNum);

    if(it != groundTruth.end())
    {
        FrameObjects tmpFrameObj = it->second;
        for(unsigned int i=0 ; i < tmpFrameObj.getObjs().size(); i++)
        {
            if(tmpFrameObj.getObjs().at(i).getCategory() != PANEL_CATEGORY )
                continue;
            cv::Rect tmpBorder = tmpFrameObj.getObjs().at(i).getBorder();
            resizeAntRecFrom1080(tmpBorder);
            result.push_back((double)countNonZero(binMask(tmpBorder))/tmpBorder.area());
            cv::rectangle(binMask, tmpBorder, cv::Scalar(0, 255, 0));
        }
    }
}

void edgeAwareSmooth(cv::Mat &img, cv::Mat &dst)
{
    // change depth
    img.convertTo(img, CV_64FC3, 1.0 / 255.0);
    // Parameter set
    const double sigma_s = DOMAIN_SIGMA_S;
    const double sigma_r = DOMAIN_SIGMA_R  ;
    const int    maxiter = DOMAIN_MAX_ITER;
    // Call domain transform filter
    domainTransformFilter(img, dst, img, sigma_s, sigma_r, maxiter);
}

void cropGroundTruth(cv::VideoCapture cap, path inFile, path outDir)
{
    /* //sample for find
    map<int, FrameObjects>::iterator it = vidObjects.find(139);
    if(it != vidObjects.end())
    {
        FrameObjects tmpFrameObj = it->second;
        for(int i=0 ; i < tmpFrameObj.getObjs().size(); i++)
        {
            cout<< tmpFrameObj.getObjs().at(i).getName() <<  endl;
        }
    }
    */
    
    /* //sample for traverse
    for (map<int , FrameObjects>::iterator it=vidObjects.begin();
         it!= vidObjects.end(); ++it)
    {
        cout << "Frame : " << it->first << endl;
        FrameObjects tmpFrameObj = it->second;
        
        for(int i=0 ; i < tmpFrameObj.getObjs().size(); i++)
        {
            cout<< tmpFrameObj.getObjs().at(i).getName() <<  endl;
        }
    }
    */
    
    map<int, FrameObjects> vidObjects;
    parseFile(inFile, vidObjects);
    cv::Mat currentframe;
    int frameCount = 0;
    while (true) {
        cap >> currentframe;
        if(currentframe.size().area() <= 0)
            break;
        
        map<int, FrameObjects>::iterator it = vidObjects.find(frameCount);
        if(it != vidObjects.end())
        {
            FrameObjects tmpFrameObj = it->second;
            for(unsigned int i=0 ; i < tmpFrameObj.getObjs().size(); i++)
            {
                cv::Rect tmpBorder(tmpFrameObj.getObjs().at(i).getBorder());
                //Scale ROI! annotation is done in 720p, the input is 1080p
                cv::Rect resizedBorder(tmpBorder.x*1.5, tmpBorder.y*1.5, tmpBorder.width*1.5, tmpBorder.height*1.5);
                cv::Rect imgBounds(0,0,currentframe.cols, currentframe.rows);
                resizedBorder = resizedBorder & imgBounds;
                
                imwrite(outDir.string() + "/" + tmpFrameObj.getObjs().at(i).getName() + "_" +
                        to_string(frameCount) + ".png", currentframe(resizedBorder));
                //rectangle(currentframe, tmpBorder, Scalar(0,0,255));
            }
        }
        //imshow("Orig" , currentframe);
        if(cvWaitKey(10) == 'q')
            break;
        frameCount++;
    }

    //Handle Memory
    currentframe.release();
    cap.release();
    //Free to go
}

void parseFile(path inFile, map<int, FrameObjects>& theMap)
{
    //Open the merged annotation file and parse to map
    ifstream txtFile(inFile.string());
    if(txtFile.is_open())
    {
        string tmpString;
        while(getline(txtFile, tmpString))
        {
            FrameObjects tmpFrameObject;
            tmpFrameObject.parse(tmpString);
            theMap[tmpFrameObject.getFrameNumber()] = tmpFrameObject;
        }
        txtFile.close();
    }
}


/*
void nkhTest()
{
    long fpsSum = 0, counter = 0 ;
    int iLowH = 68;
    int iHighH = 93;
    
    int iLowS = 40;
    int iHighS = 255;
    
    int iLowV = 47;
    int iHighV = 174;
    
    int iLowB = 68;
    int iHighB = 93;
    
    int iLowG = 63;
    int iHighG = 255;
    
    int iLowR = 55;
    int iHighR = 174;
    
    int hueValue = 255;
    
    Mat src, dst, imgHSV;
    Mat dst2 ;
    VideoCapture cap("../test/35.mp4");
    initFPSTimer();
    while(true)
    {
        fpsCalcStart();
        cap>>src ;
        if( src.size().area() <=0 )
        {
            break;
        }
        
        
        //        resize(src,src, cvSize(src.size().width/2,src.size().height/2));
        cvtColor(src, imgHSV, COLOR_BGR2HSV);
        
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), dst); //Threshold the image
        
        //        inRange(src, Scalar(iLowB, iLowG, iLowR), Scalar(iHighB, iHighG, iHighR), dst2); //Threshold the image
        
        
        //morphological opening (remove small objects from the foreground)
        erode(dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(5,5)) );
        dilate( dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(5,5)) );
        //morphological closing (fill small holes in the foreground)
        dilate( dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(5,5)) );
        erode(dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(5,5)) );
        
        //        Mat resized;
        //resize(dst,dst, cvSize(dst.size().width/2.0,dst.size().height/2.0));
        
        medianBlur(dst,dst,5);
        
        //nkhImshow_Write(src, dst, "HSV", "HSV","green");
        //        nkhImshow_Write(src, dst2, "RGB", "RGB","green");
        
        vector<vector<Point> > contours;
        vector<Vec4i> order;
        GaussianBlur( dst, dst, Size(5, 5), 1,1 );
        Mat nullImg ;
        int params[]={70,210,3};
        Mat edges;
        blur( dst, edges, Size(3,3) );
        
        /// TIME CONSUMING
        //Canny( dst, edges, params[0], params[1],params[2]);
        
        
        findContours( edges, contours, order, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        
        Mat showContours = Mat::zeros( edges.size(), CV_8UC3 );
        
        /// TIME BOTTLENECK : Extra?
        //        resize(src,showContours,cvSize(src.size().width/2.0,src.size().height/2.0));
        
        
        RNG rng(12345);
        
        vector<vector<Point> > contoursApprox;
        contoursApprox.resize(contours.size());
        for( size_t k = 0; k < contours.size(); k++ )
        {
            double epsilon = 0.05*arcLength(contours[k],true);
            approxPolyDP(Mat(contours[k]), contoursApprox[k], epsilon, true);
        }
        
        vector<Moments> mu(contoursApprox.size() );
        for( int i = 0; i < contoursApprox.size(); i++ )
        { mu[i] = moments( contoursApprox[i], false ); }
        
        
        vector<Point2f> mc( contoursApprox.size() );
        for( int i = 0; i < contoursApprox.size(); i++ )
        { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }
        
        vector<RotatedRect> minRect( contours.size() );
        for( int i = 0; i< contoursApprox.size(); i++ )
        {
            long long carea = contourArea(contoursApprox[i]) ;
            long long area = dst.size().width*dst.size().height;
            minRect[i] = minAreaRect( Mat(contoursApprox[i]) );
            if( carea< (area)/900.0 || carea > (area)/3.0
                    || (int)mc[i].y > dst.size().height/2  || minRect[i].size.width < minRect[i].size.height
                    || (minRect[i].angle <-30 ))
                continue;
                
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            //            drawContours( showContours, contoursApprox, i, color, 2, 8, order, 0, Point() );
            Point2f rect_points[4]; minRect[i].points( rect_points );
            for( int j = 0; j < 4; j++ )
                line( showContours, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
        }
        
        // TIME CONSUMING
        //imshow("Contoures", showContours);
        
        //        for( int i = 0; i< contours.size(); i++ )
        //        {
        //            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        //            drawContours( showContours, contours, i, color, 2, 8, order, 0, Point() );
        //        }
        //        imshow("Contoures",showContours);
        // TIME CONSUMING
        // imshow("Thresh",dst);
        
        fpsCalcEnd();
        if(timerFPS !=INFINITY )
        {
            fpsSum += timerFPS;
            //cout << counter << " " << timerFPS << endl ;
        }
        counter++;
        cout << timerFPS << endl;
        if (waitKey(10) == 'q') //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            break;
        }
    }
}
*/
