#include <QCoreApplication>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
using namespace boost::accumulators;
#include <boost/filesystem.hpp>
using namespace boost::filesystem;

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <sys/time.h>
using namespace std;

#include <fstream>
#include <map>

#include "nkhUtil.h"
#include "nkhAlgorithms.h"
#include "egbiscv.h"


/*----------------- nkhEnd: global vars and defs -----------------*/

//void nkhTest();
void cropGroundTruth(cv::VideoCapture cap, path inFile, path outDir);

void parseFile(path inFile, map<int, FrameObjects>& outMap);

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
void nkhSeg(cv::Ptr<nkhGraphSegmentationImpl> segPtr, cv::Mat& img, cv::Mat& cutMatrix, cv::Mat& egbisSeg)
{
    segPtr->processImage(img, cutMatrix, egbisSeg);
}



typedef unsigned long long timestamp_t;

static timestamp_t
get_timestamp ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}



float sigma_egbis ;
float k_egbis ;
int min_size_egbis ;
int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);

    if (WITH_CUDA == 1) {
//        cerr << "USING CUDA\n";
//        cv::gpu::setDevice(0); //not neede with opencv
    } else {
//        cerr << "NOT USING CUDA\n";
    }

#ifdef _OPENMP
//    cout << "USINGOPENMP\n";
#endif

    if (argc == 7)
    {
        path inVid(argv[1]), inFile(argv[2]), outDir(argv[3]);
        checkRegularFile(inVid);
        checkRegularFile(inFile);
        checkDir(outDir);
        sigma_egbis = atof(argv[4]);
        k_egbis = atof(argv[5]);
        min_size_egbis = atoi(argv[6]);
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

    cv::setUseOptimized(true);
    cv::setNumThreads(16);
    cv::VideoCapture cap(inVid.string());
    
    //Task 1.5: open the map for evaluation
    map<int, FrameObjects> groundTruth;
    parseFile(inFile, groundTruth);

    cv::Mat currentFrame;
    int frameCount = 0;
    initFPSTimer();
    vector<double> evalColmask, evalNegColmask;

//    cropGroundTruth(cap, inFile, outDir);
//    cap.release();
//    return;

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

    cv::Mat prevFrame_gray2, flowImg;

    int seg_sum = 0 , mser_sum = 0 ;
    long double fps_sum = 0, time_g = 0 , time_w = 0 , time_b = 0;

    while(true)
    {
        cap >> currentFrame;
        if(currentFrame.size().area() <= 0)
        {
//            cap.set(CV_CAP_PROP_POS_AVI_RATIO , 0);
//            continue;
            break;
        }
        fpsCalcStart();
        cv::Mat frameResized, frameResizedHalf_gray,frameResized_gray2, frameResizedHalf, frameResized2;
        cv::resize(currentFrame, frameResized, cv::Size(currentFrame.size().width/RESIZE_FACTOR,
                                             currentFrame.size().height/RESIZE_FACTOR));

        cv::resize(currentFrame, frameResized2, cv::Size(currentFrame.size().width/RESIZE_FACTOR2,
                                             currentFrame.size().height/RESIZE_FACTOR2));

        cv::resize(currentFrame, frameResizedHalf, cv::Size(currentFrame.size().width/2,
                                             currentFrame.size().height/2));
        cv::cvtColor(frameResizedHalf, frameResizedHalf_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frameResized2, frameResized_gray2, cv::COLOR_BGR2GRAY);


        if(prevFrame_gray2.empty())
        {
            prevFrame_gray2 = frameResized_gray2.clone();
            continue;
        }
        //TODO: implement with GPU
        //Mat edgeSmooth = measure<std::chrono::milliseconds>(edgeAwareSmooth, frameResized);
        cv::Mat edgeSmooth, edgeSmooth_gray, edgeSmooth_resize2, edgeSmooth_Half;

        dtfWrapper( frameResized, edgeSmooth);
        //dtfWrapper(frameResizedHalf, edgeSmooth_Half);
        cv::blur(frameResizedHalf, edgeSmooth_Half, Size(21,21));
        cv::Mat edgeSmoothLow, edgeSmoothLow_gray;
        cv::ximgproc::dtFilter(frameResized, frameResized, edgeSmoothLow, 20, 140, cv::ximgproc::DTF_RF); //r 350. 50. nc


        cv::cvtColor(edgeSmooth, edgeSmooth_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(edgeSmoothLow, edgeSmoothLow_gray, cv::COLOR_BGR2GRAY);

//        edgeSmooth = colorReduce(edgeSmooth, 64);
//        edgeSmoothLow = colorReduce(edgeSmoothLow, 64);

        cv::resize(edgeSmoothLow, edgeSmooth_resize2, cv::Size(currentFrame.size().width/RESIZE_FACTOR2,

                                                               currentFrame.size().height/RESIZE_FACTOR2));


        cv::Mat hsvFrameRes2, labFrameRes2, hlsFrameRes2, hsvResized, saliencyV;
        cv::cvtColor(frameResized2, hsvFrameRes2, cv::COLOR_BGR2HSV);
        cv::cvtColor(frameResized2, hsvResized, cv::COLOR_BGR2HLS);
        cv::cvtColor(frameResized2, labFrameRes2, cv::COLOR_BGR2Lab);
        cv::cvtColor(frameResized2, hlsFrameRes2, cv::COLOR_BGR2HLS);
        cv::Mat chans[3];
        cv::split(hsvResized, chans);
        //SaliencyMap
        cv::Mat saliency, saliency72, saliencyOrig, saliency72Orig;
        computeSaliency(frameResized, 55 , saliency);

        computeSaliency(frameResized, 55 , saliency72);


        timestamp_t t0 = get_timestamp();
        timestamp_t t1 = get_timestamp();
        double secs = (t1 - t0) / 1000000.0L;
//        cout << "W1: " << secs << endl;
//        time_w += secs;

         computeSaliency(edgeSmoothLow, 64 , saliencyV);

        saliencyOrig = saliency.clone();
        saliency72Orig = saliency72.clone();

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

        whiteThresh2(frameResized, saliency72, whiteMask);


        cv::Mat greenMask;

//        t0 = get_timestamp();
        greenThresh1(edgeSmoothLow, greenMask);
//        t1 = get_timestamp();
//        secs = (t1 - t0) / 1000000.0L;
//        cout << "G1: " << secs << endl;
//        time_g += secs;

        cv::Mat brownMask;
        brownThresh1(edgeSmoothLow, brownMask);
//        t0 = get_timestamp();
//        t1 = get_timestamp();
//        secs = (t1 - t0) / 1000000.0L;
//        cout << "B1: " << secs << endl;
//        time_b += secs;

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

        /*
//        int erosionDilation_size = 3;
//        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
//                                                    cv::Size(2*erosionDilation_size + 1,
//                                                             2*erosionDilation_size+1));
*/
        cv::Mat colorMask = whiteMask | greenMask |  brownMask;
//        */

        /*
        cv::blur(colorMask, colorMask, cv::Size(12,12));
        cv::erode(colorMask, colorMask, element);
        cv::dilate(colorMask, colorMask, element);
        */

//        /*
        frameResized.copyTo(masked, colorMask);
        cv::Mat twoThird(colorMask);// = cv::Mat::zeros(edgeSmooth.rows, edgeSmooth.cols, edgeSmooth.type());
        twoThird(cv::Rect(0,2*twoThird.size().height/3.0,
                          twoThird.size().width, twoThird.size().height/3.0)) = cv::Scalar::all(0);
        cv::Mat tmp;
        masked.copyTo(tmp, twoThird);
        cv::Mat saliency32;
        computeSaliency(tmp, 32, saliency32);
        saliency32 *= 3;
        saliency32.convertTo( saliency32, CV_8U );
        threshold( saliency32, binMask, 0, 255, cv::THRESH_BINARY );
        fin = colorMask & twoThird ;//& binMask;
        masked.release();
        tmp.copyTo(masked, fin);
        cv::Mat edgeMasked(edgeSmoothLow);
        edgeSmooth.copyTo(edgeMasked, fin);
//        */



        //MSER
/*

        std::vector< std::vector< cv::Point> > contours;
        std::vector< cv::Rect> bboxes;
        // Ptr< MSER> mser = MSER::create(21, (int)(0.00002*textImg.cols*textImg.rows),
        //(int)(0.05*textImg.cols*textImg.rows), 1, 0.7);
        t0 = get_timestamp();
        cv::Ptr< cv::MSER> mser = cv::MSER::create();//5, 200, frameResized.size().area()/3.0, 0.01);
        mser->detectRegions(frameResized, contours, bboxes);
        cv::Mat tmpMserMask = cv::Mat::zeros(masked.rows, masked.cols, fin.type());

        for (vector<cv::Point> v : contours){
//            Vec3b color(rand()%255,rand()%255,rand()%255);
                for (cv::Point p : v){
                    tmpMserMask.at<uchar>(p.y, p.x) = 255;
//                    frameResized.at<Vec3b>(Point(p.x, p.y)) = color;
                }
            }
        tmpMserMask &= twoThird;
//        for (int i = 0; i < bboxes.size(); i++)
//        {
//            char name[20];
//            sprintf(name, "/mser-f%d", frameCount);
//            resizeDetRecTo1080(bboxes[i]);
//            imwrite(outDir.string() + name + "n" + to_string(i) +".png", currentFrame(bboxes[i]));
//        }


//        maybeImshow("mserbb", tmpMserMask);
        mser_sum += contours.size();
        t1 = get_timestamp();
        secs = (t1 - t0) / 1000000.0L;
//        cout << "B1: " << secs << endl;
        time_b += secs;
//        cout << "MSERS " << contours.size() << endl;
*/

        //contour appx

/*
        cv::Mat grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;

        /// Gradient X
        //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
        cv::Sobel( edgeSmooth_gray, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT );
        cv::convertScaleAbs( grad_x, abs_grad_x );
        cv::Sobel( edgeSmooth_gray, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT );
        cv::convertScaleAbs( grad_y, abs_grad_y );
        cv::addWeighted(abs_grad_y, 0.5, abs_grad_x, 0.5, 0, abs_grad_x);
//        saliency32 *=20 ;
//        cv::addWeighted(saliency, 0.7, abs_grad_x, 0.5, 0, abs_grad_x);
//        abs_grad_x &= colorMask;
        cv::threshold(abs_grad_x, abs_grad_x, 30, 255, cv::THRESH_BINARY);
        //cv::ximgproc::dtFilter(abs_grad_x, abs_grad_x, abs_grad_x, 70, 70, cv::ximgproc::DTF_RF); //r 350. 50. nc
*/

        /*
        cv::Mat edgeImg;
        cv::cuda::GpuMat devSmooth(frameResized2), devEqHist, devGraySmooth(frameResized_gray2), devEdgeImg;
//        cv::cuda::cvtColor(devSmooth, devGraySmooth, cv::COLOR_BGR2GRAY);
//        cv::cuda::equalizeHist(devGraySmooth, devEqHist);
        cv::Ptr<cv::cuda::Filter> laplacPtr = cv::cuda::createLaplacianFilter(devGraySmooth.type(), devGraySmooth.type(),
                                                                              3, 1);
        laplacPtr->apply(devGraySmooth, devEdgeImg);
//        cv::Ptr<cv::cuda::CannyEdgeDetector> cudaCanny = cv::cuda::createCannyEdgeDetector(100, 200);
//        cudaCanny->detect(devGraySmooth, devEdgeImg);
        //        /*
//        int erosionDilation_size = 3;
//        Mat element = cv::getStructuringElement(MORPH_RECT, Size(2*erosionDilation_size + 1,    2*erosionDilation_size+1));
//        Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_ERODE, devEdgeImg.type(), element);
//        dilateFilter->apply(devEdgeImg, devEdgeImg);
        devEdgeImg.download(edgeImg);

         */

        /*
        //        edgeImg &= fin;
        //cv::ximgproc::dtFilter(edgeImg, edgeImg, edgeImg, 80, 190, cv::ximgproc::DTF_RF); //r 350. 50. nc

        //
        */

        //Visualize

        //OpticalFlow
        /*
//        cv::cuda::GpuMat devPrev(prevFrame_gray2), devCurr(frameResized_gray2), devFlow;
//        cv::Ptr <cv::cuda::FarnebackOpticalFlow> optflowPtr = cv::cuda::FarnebackOpticalFlow::create(2, 0.5, true, 10, 3,5, 1.2);
        cv::calcOpticalFlowFarneback(prevFrame_gray2, frameResized_gray2, flowImg, 0.5, 3, 15, 3, 5, 1.2,0);
//        optflowPtr->calc(devPrev, devCurr, devFlow);


        cv::Mat cflow;
//        devFlow.download(cflow);
        cvtColor(prevFrame_gray2, cflow, CV_GRAY2BGR);

        drawOptFlowMap(flowImg, cflow, 2, CV_RGB(0, 255, 0));
*/
        /*
        cv::resize(fin, fin, frameResizedHalf.size());
        cv::Mat maskedHalf;
        frameResizedHalf.copyTo(maskedHalf, fin);
        */

        cv::Mat egbisImage, tmpOut;
        cv::Mat currTest(hlsFrameRes2);
        currTest.copyTo(tmpOut);
//        cv::cvtColor(tmpOut, labFrameRes2, cv::COLOR_Lab2BGR);
        int minSegSize = tmpOut.size().area()/200;

        cv::Mat egbisSeg, grayImg;
        /*
//        cvtColor(frameResized2, grayImg, COLOR_BGR2GRAY);
//        Mat grad_x, grad_y;
//        Mat abs_grad_x, abs_grad_y;

//        int scale = 1;
//        int delta = 0;
//        int ddepth = CV_16S;
//        /// Gradient X
//        Sobel( grayImg, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
//        convertScaleAbs( grad_x, abs_grad_x );
//        /// Gradient Y
//        Sobel( grayImg, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
//        convertScaleAbs( grad_y, abs_grad_y );
//        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edgeImg );
*/
        Ptr<nkhGraphSegmentationImpl> segPtr = createGraphSegmentation(sigma_egbis, k_egbis, min_size_egbis);
//        measure<chrono::milliseconds>(nkhSeg, segPtr, edgeSmoothLow, edgeImg, egbisSeg);
//        measure<chrono::milliseconds>(egbisVisualise, egbisSeg, egbisImage);
        cv::Mat ensembleFrame, hlsSmooth, hsvSmooth, hlsChans[3], hsvChans[3], labSmooth, costMatrixes[3], costMatrix;
        /*
//        cv::cvtColor(edgeSmoothLow, hlsSmooth, cv::COLOR_BGR2HLS);
//        cv::cvtColor(edgeSmoothLow, hsvSmooth, cv::COLOR_BGR2HSV);
//        cv::cvtColor(edgeSmoothLow, labSmooth, cv::COLOR_BGR2Lab);
//        cv::split(hlsSmooth, hlsChans);
//        cv::split(hsvSmooth, hsvChans);
//        hlsChans[2] = hsvChans[2].clone();
//        cv::merge(hlsChans, 3 , ensembleFrame);

//        costMatrixes[0]=whiteMask.clone();
//        costMatrixes[1]=greenMask.clone();
//        costMatrixes[2]=brownMask.clone();
//        cv::merge(costMatrixes, 3, costMatrix);
*/
        nkhSeg(segPtr, edgeSmooth, costMatrix, egbisSeg);
        int nb_segs = egbisVisualise(egbisSeg, egbisImage);
        seg_sum += nb_segs;
//        /*
        cv::Mat tmpEgbMask = cv::Mat::zeros(egbisImage.rows, egbisImage.cols, fin.type());
         cv::Mat toWrite = frameResized.clone(), edgTowrite(edgeSmooth);
        for (int i = 0 ; i <= nb_segs ; i++)
        {
            cv::Mat s1;
            cv::inRange(egbisSeg, cv::Scalar(i), cv::Scalar(i), s1);
//            imshow("t1" , s1&fin);
//            cvWaitKey(10);

            if(cv::countNonZero(s1&fin) < 0.2*cv::countNonZero(s1))
                continue;
            tmpEgbMask |= s1;
            char name[20];
            sprintf(name, "f%d-%d", frameCount, i);

            vector<vector<Point> > contours;
             vector<Vec4i> hierarchy;
             /// Detect edges using Threshold             /// Find contours
             findContours( s1, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
             /// Approximate contours to polygons + get bounding rects and circles
             vector<Rect> boundRect( contours.size() );
             for( int i = 0; i < contours.size(); i++ )
                {
                  boundRect[i] = boundingRect( contours[i] );
                  frameResized.copyTo(edgTowrite,s1);
                  imwrite(outDir.string() + "/" + name + "c" + to_string(i) +".png", edgTowrite(boundRect[i]));
                  Scalar color = Scalar( rand()%255, rand()%255, rand()%255 );
                  rectangle( toWrite, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
                }
//             /// Draw polygonal contour + bonding rects + circles
//             for( int i = 0; i< contours.size(); i++ )
//                {
////                  Scalar color = Scalar( 0,255,255 );
////                  drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
//                    frameResizedHalf.copyTo(toWrite, s1);
////                  rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
//                }

        }
//        */
/*
//         cv::cuda::Stream stream[3];
//         for(int i=0; i<3; i++){
//             for(int j=0; j<3; j++){
//                 for(int k=0; k<4; k++){
//                     cv::gpu::multiply(pyramid[i][j][k], weightmap[i][k], pyramid[i][j][k], stream[i]);
//                 }
//             }
//         }
*/
/*
        char controlChar = maybeImshow("Orig", masked, 30) ;

        controlChar = maybeImshow("Saliency", frameResized) ;
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
*/

        //Eval
        evaluateMasked(tmpEgbMask, groundTruth, frameCount, evalColmask);
        evaluateNonMasked(tmpEgbMask, groundTruth, frameCount, evalNegColmask);

                     maybeImshow("cont", toWrite, 30) ;

        imwrite(outDir.string() + "/" + "frame-" + to_string(frameCount) +".png", toWrite);
        fpsCalcEnd();
//        cout<< timerFPS  << "fps" << endl;
        if(timerFPS != INFINITY)
            fps_sum += timerFPS;
        frameCount++;
        prevFrame_gray2 = frameResized_gray2.clone();
    }
    calcMeanVar(evalColmask, "egbis", false);
    calcMeanVar(evalNegColmask, "egbis-neg");


//    cout << "Segs " << (double)seg_sum/frameCount << endl;
//    cout << "MSERs " << (double)mser_sum/frameCount << endl;
//    cout << "FPS " << (double)fps_sum/frameCount << endl;
//    cout << "W " << (double)time_w/frameCount << endl;
//    cout << "G " << (double)time_g/frameCount << endl;
//    cout << "B " << (double)time_b/frameCount << endl;
    //Handle Memory
    cap.release();
    currentFrame.release();
    groundTruth.clear();
    evalColmask.clear();
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
void evaluateNonMasked(cv::Mat& binMask, map<int, FrameObjects>& groundTruth, int frameNum, vector<double>& result) //binMask should be binary
{
    cv::Mat tmp=binMask.clone();
    tmp.setTo(255);
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
            tmp(tmpBorder) = 0 ;
            cv::rectangle(binMask, tmpBorder, cv::Scalar(0, 0, 0));
        }
        tmp = tmp & binMask;
//        maybeImshow("neg", tmp);
        //maybeImshow("bin", binMask);
        result.push_back((double)countNonZero(tmp)/binMask.size().area());
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
                //New-extend
                double extendFactor = 0.8;
                cv::Rect resizedBorder(tmpBorder.x*1.5 - extendFactor*tmpBorder.width,
                                       tmpBorder.y*1.5 - extendFactor*tmpBorder.height,
                                       tmpBorder.width*1.5 + 2*extendFactor*tmpBorder.width,
                                       tmpBorder.height*1.5 + 2*extendFactor*tmpBorder.height);
//                cv::Rect resizedBorder(rand()%700 , rand()%600,
//                                       300, 300);


                cv::Rect imgBounds(0,0,currentframe.cols, currentframe.rows);
                resizedBorder = resizedBorder & imgBounds;
                
                imwrite(outDir.string() + "/" + tmpFrameObj.getObjs().at(i).getName() + "_" +
                        to_string(frameCount) + ".png", currentframe(resizedBorder));
                //rectangle(currentframe, tmpBorder, Scalar(0,0,255));
            }
        }
        else
        {
//            cv::Rect resizedBorder(0 , 350,
//                                   500, 500);

//            cv::Rect imgBounds(0,0,currentframe.cols, currentframe.rows);
//            resizedBorder = resizedBorder & imgBounds;

//            imwrite(outDir.string() + "/" + "else" + "_" +
//                    to_string(frameCount) + ".png", currentframe(resizedBorder));
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
