#include <QCoreApplication>
#include <opencv/cv.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/cudaarithm.hpp>
//#include <opencv2/cudawarping.hpp>

using namespace cv;
#include <fstream>
#include <map>
using namespace std;

#include "nkhUtil.h"
#include "FrameObjects.h"

/****************** nkhStart: global vars and defs ******************/
#define RESIZE_FACTOR 5
#define DOMAIN_SIGMA_S 37.0
#define DOMAIN_SIGMA_R 1.8
#define DOMAIN_MAX_ITER 3
/****************** nkhEnd: global vars and defs ******************/

void nkhMain(path inVid, path inFile, path outDir);

/************************* nkhStart: FPS counter *************************/
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
/************************* nkhEnd: FPS counter *************************/

/******************** nkhStart opencv wrappers ********************/
void nkhImshow(const char* windowName, cv::Mat img)
{
    imshow(windowName, img);
}

char maybeImshow(const char* windowName, cv::Mat img, int waitKeyTime=10)
{
    if(WITH_VISUALIZATION)
    {
        nkhImshow(windowName, img);
        return cvWaitKey(waitKeyTime);
    }
    return 0;
}

/******************** nkhEnd opencv wrappers ********************/

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
}
/******************** nkhEnd domainTransform ********************/

/******************** nkhStart Saliency ********************/
void computeSaliency(cv::Mat imgGray, cv::Mat& saliencyMap)
{
    Mat grayDown;
    std::vector<Mat> mv;
    Size resizedImageSize( 64, 64 );
    
    Mat realImage( resizedImageSize, CV_64F );
    Mat imaginaryImage( resizedImageSize, CV_64F );
    imaginaryImage.setTo( 0 );
    Mat combinedImage( resizedImageSize, CV_64FC2 );
    Mat imageDFT;
    Mat logAmplitude;
    Mat angle( resizedImageSize, CV_64F );
    Mat magnitude( resizedImageSize, CV_64F );
    Mat logAmplitude_blur, imageGR;
    
    if( imgGray.channels() == 3 )
    {
        cvtColor( imgGray, imageGR, COLOR_BGR2GRAY );
        resize( imageGR, grayDown, resizedImageSize, 0, 0, INTER_LINEAR );
    }
    else
    {
        resize( imgGray, grayDown, resizedImageSize, 0, 0, INTER_LINEAR );
    }
    
    grayDown.convertTo( realImage, CV_64F );
    
    mv.push_back( realImage );
    mv.push_back( imaginaryImage );
    merge( mv, combinedImage );
    dft( combinedImage, imageDFT );
    split( imageDFT, mv );
    
    //-- Get magnitude and phase of frequency spectrum --//
    cartToPolar( mv.at( 0 ), mv.at( 1 ), magnitude, angle, false );
    log( magnitude, logAmplitude );
    //-- Blur log amplitude with averaging filter --//
    blur( logAmplitude, logAmplitude_blur, Size( 3, 3 ), Point( -1, -1 ), BORDER_DEFAULT );
    
    exp( logAmplitude - logAmplitude_blur, magnitude );
    //-- Back to cartesian frequency domain --//
    polarToCart( magnitude, angle, mv.at( 0 ), mv.at( 1 ), false );
    merge( mv, imageDFT );
    dft( imageDFT, combinedImage, DFT_INVERSE );
    split( combinedImage, mv );
    
    cartToPolar( mv.at( 0 ), mv.at( 1 ), magnitude, angle, false );
    GaussianBlur( magnitude, magnitude, Size( 5, 5 ), 8, 0, BORDER_DEFAULT );
    magnitude = magnitude.mul( magnitude );
    
    double minVal, maxVal;
    minMaxLoc( magnitude, &minVal, &maxVal );
    
    magnitude = magnitude / maxVal;
    magnitude.convertTo( magnitude, CV_32F );
    
    resize( magnitude, saliencyMap, imgGray.size(), 0, 0, INTER_LINEAR );
    
    // visualize saliency map
    //imshow( "Saliency Map Interna", saliencyMap );

}

/******************** nkhEnd Saliency ********************/
void edgeAwareSmooth(cv::Mat img, cv::Mat& dst);

//void nkhTest();
void cropGroundTruth(VideoCapture cap, path inFile, path outDir);

map<int, FrameObjects> parseFile(path inFile);

int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);
    
    if (WITH_CUDA == 1) {
        cerr << "USING CUDA\n";
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
    VideoCapture cap(inVid.string());
    
    //TODO: parse commandline arguments for different tasks
    //task1 : cropGroundTruth(VideoCapture cap, path inFile, path outDir)
    //task1 is done, ran once.
    
    //task2: resize and preproc.
    Mat currentFrame;
    int frameCount = 0;
    initFPSTimer();
    while(true)
    {
        cap >> currentFrame;
        if(currentFrame.size().area() <= 0)
            break;
        fpsCalcStart();
        
        Mat frameResized, frameResized_gray;
        resize(currentFrame, frameResized, Size(currentFrame.size().width/RESIZE_FACTOR,
                                             currentFrame.size().height/RESIZE_FACTOR));
        cvtColor(frameResized, frameResized_gray, COLOR_BGR2GRAY);

        //SaliencyMap
        Mat saliency;
        //
        //Mat edgeSmooth;
        //edgeAwareSmooth(frameResized, edgeSmooth);
        //edgeSmooth.convertTo(edgeSmooth, CV_32F);
        //cvtColor(edgeSmooth, frameResized_gray, COLOR_BGR2GRAY);
        //
        //blur(frameResized_gray, frameResized_gray, Size(10,10));
        computeSaliency(frameResized_gray, saliency);
        Mat masked, binMask;
        saliency = saliency * 10;
        saliency.convertTo( saliency, CV_8U );
        // adaptative thresholding using Otsu's method, to make saliency map binary
        threshold( saliency, binMask, 0, 255, THRESH_BINARY | THRESH_OTSU );
        frameResized.copyTo(masked, binMask);
        //maybeImshow("orig", frameResized);
        if(maybeImshow("Saliency", masked) == 'q') break;

        /*
        if(maybeImshow("resized orig", frameResized)=='q')
            break;
            */
        //TODO: implement with GPU
        //Mat edgeSmooth = measure<std::chrono::milliseconds>(edgeAwareSmooth, frameResized);
        /*
        Mat edgeSmooth = edgeAwareSmooth(frameResized);
        if(maybeImshow("Smooth", edgeSmooth)=='q')
            break;
        */

        //threshold

        //evaluate Correlation
        
        //contour appx
        
        fpsCalcEnd();
        cout<< timerFPS << endl;
        frameCount++;
    }
    return;
}

void edgeAwareSmooth(cv::Mat img, Mat &dst)
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

void cropGroundTruth(VideoCapture cap, path inFile, path outDir)
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
    
    map<int, FrameObjects> vidObjects = parseFile(inFile);
    Mat currentframe;
    int frameCount = 0;
    while (true) {
        cap >> currentframe;
        if(currentframe.size().area() <= 0)
            break;
        
        map<int, FrameObjects>::iterator it = vidObjects.find(frameCount);
        if(it != vidObjects.end())
        {
            FrameObjects tmpFrameObj = it->second;
            for(int i=0 ; i < tmpFrameObj.getObjs().size(); i++)
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
}

map<int, FrameObjects> parseFile(path inFile)
{
    map<int, FrameObjects> theMap;
    
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
    return theMap;
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
