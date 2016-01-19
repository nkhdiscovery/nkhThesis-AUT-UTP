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
using namespace std;

#include <fstream>
#include <map>

//#include "nkhUtil.h"
#include "nkhAlgorithms.h"


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


int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);

    if (WITH_CUDA == 1) {
        cerr << "USING CUDA\n";
//        cv::gpu::setDevice(0); //not neede with opencv
    } else {
        cerr << "NOT USING CUDA\n";
    }

#ifdef _OPENMP
    cout << "USINGOPENMP\n";
#endif

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

    cv::setUseOptimized(true);
    cv::setNumThreads(16);

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

    cv::Mat prevFrame_gray2, flowImg;

    while(true)
    {
        cap >> currentFrame;
        if(currentFrame.size().area() <= 0)
        {
            cap.set(CV_CAP_PROP_POS_AVI_RATIO , 0);
            continue;
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
        cv::Mat edgeSmooth, edgeSmooth_gray, edgeSmooth_resize2;

        dtfWrapper(frameResized, edgeSmooth);

        cv::Mat edgeSmoothLow, edgeSmoothLow_gray;
        cv::ximgproc::dtFilter(frameResized, frameResized, edgeSmoothLow, 80, 190, cv::ximgproc::DTF_RF); //r 350. 50. nc


        cv::cvtColor(edgeSmooth, edgeSmooth_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(edgeSmoothLow, edgeSmoothLow_gray, cv::COLOR_BGR2GRAY);

        edgeSmooth = colorReduce(edgeSmooth, 64);
        edgeSmoothLow = colorReduce(edgeSmoothLow, 64);

        cv::resize(edgeSmoothLow, edgeSmooth_resize2, cv::Size(currentFrame.size().width/RESIZE_FACTOR2,
                                                        currentFrame.size().height/RESIZE_FACTOR2));


        //SaliencyMap
        cv::Mat saliency, saliency72, saliencyOrig, saliency72Orig;
        computeSaliency(frameResized, 55 , saliency);
        computeSaliency(frameResized, 64 , saliency72);
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

        /*
        int erosionDilation_size = 3;
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                                    cv::Size(2*erosionDilation_size + 1,
                                                             2*erosionDilation_size+1));

        cv::Mat colorMask = whiteMask | greenMask | brownMask;
        */

        /*
        cv::blur(colorMask, colorMask, cv::Size(12,12));
        cv::erode(colorMask, colorMask, element);
        cv::dilate(colorMask, colorMask, element);
        */

        /*
        frameResized.copyTo(masked, colorMask);
        cv::Mat twoThird(colorMask);// = cv::Mat::zeros(edgeSmooth.rows, edgeSmooth.cols, edgeSmooth.type());
        twoThird(cv::Rect(0,2*twoThird.size().height/3.0, twoThird.size().width, twoThird.size().height/3.0)) = cv::Scalar::all(0);
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
        */


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
        cv::cuda::GpuMat devSmooth(edgeSmoothLow), devEqHist, devGraySmooth(edgeSmoothLow_gray), devEdgeImg;
//        cv::cuda::cvtColor(devSmooth, devGraySmooth, cv::COLOR_BGR2GRAY);
//        cv::cuda::equalizeHist(devGraySmooth, devEqHist);
        cv::Ptr<cv::cuda::CannyEdgeDetector> cudaCanny = cv::cuda::createCannyEdgeDetector(20, 120);
        cudaCanny->detect(devGraySmooth, devEdgeImg);
        devEdgeImg.download(edgeImg);
        edgeImg &= fin;
        //cv::ximgproc::dtFilter(edgeImg, edgeImg, edgeImg, 80, 190, cv::ximgproc::DTF_RF); //r 350. 50. nc
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
        int num_ccs;

//        egbisImage = runEgbisOnMat(edgeSmooth_resize2, 0.5, 1000, 1000, &num_ccs);

        cv::Ptr<cv::ximgproc::segmentation::GraphSegmentation> gs =
                cv::ximgproc::segmentation::createGraphSegmentation(0.0, 10000, 10);
        cv::Mat hsvFrameRes2, labFrameRes2;
        cv::cvtColor(frameResized2, hsvFrameRes2, cv::COLOR_BGR2HSV);
        cv::cvtColor(frameResized2, labFrameRes2, cv::COLOR_BGR2Lab);
        cv::ximgproc::dtFilter(hsvFrameRes2, hsvFrameRes2, tmpOut,
                               80, 150, cv::ximgproc::DTF_RF); //r 350. 50. nc

//       egbisImage = runEgbisOnMat(tmpOut, 0.5, 1000, 100, &num_ccs);
//       cv::cvtColor(tmpOut, hsvFrameRes2, cv::COLOR_HSV2BGR);
//        gs->processImage(labFrameRes2, egbisImage);

        /*
        double min, max;
        cv::minMaxLoc(egbisImage, &min, &max);

        int nb_segs = (int)max + 1;

//        std::cout << nb_segs << " segments" << std::endl;

        tmpOut = cv::Mat::zeros(egbisImage.rows, egbisImage.cols, CV_8UC3);

        uint* p;
        uchar* p2;

        for (int i = 0; i < egbisImage.rows; i++) {

            p = egbisImage.ptr<uint>(i);
            p2 = tmpOut.ptr<uchar>(i);

            for (int j = 0; j < egbisImage.cols; j++) {
                cv::Scalar color = color_mapping(p[j]);
                p2[j*3] = color[0];
                p2[j*3 + 1] = color[1];
                p2[j*3 + 2] = color[2];
            }
        }
        */

        //cv::pyrMeanShiftFiltering(edgeSmooth, egbisImage, 90, 30, 1, cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 1, 1));
///*
        char controlChar = maybeImshow("edg", hsvFrameRes2) ;

//        controlChar = maybeImshow("egbis", egbisImage) ;
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
//*/
        fpsCalcEnd();
        cout<< timerFPS << endl;
        frameCount++;

        prevFrame_gray2 = frameResized_gray2.clone();

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
