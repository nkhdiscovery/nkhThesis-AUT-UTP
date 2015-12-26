#include <QCoreApplication>
#include <opencv/cv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
using namespace cv;
#include <fstream>
#include <map>
using namespace std;

#include "nkhUtil.h"
#include "FrameObjects.h"

/****************** nkhStart: global vars ******************/

/****************** nkhEnd: global vars ******************/

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

//void nkhTest();

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

    map<int, FrameObjects> vidObjects = parseFile(inFile);

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

    //Open the video file
    VideoCapture cap(inVid.string());
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
    return;
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
