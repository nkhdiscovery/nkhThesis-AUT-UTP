#include <QCoreApplication>
#include <cstdio>
#include <iostream>
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#include <vector>
#include <algorithm>
using namespace std;
#include <opencv/cv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
using namespace cv;

#include <time.h>
#include <limits.h>
#include <cstring>

#include "errorcodes.h"


/****************** nkhStart: macro utils ******************/
#define errStr(x) #x

#define checkPath(p) if(!exists(p)){\
    cerr << p << " : " << errStr(NO_SUCH_FILE_OR_DIR\n);\
    return NO_SUCH_FILE_OR_DIR; }
#define checkDir(p) checkPath(p);\
    if(!is_directory(p)){\
    cerr << p << " : " << errStr(NOT_A_DIR\n);\
    return NOT_A_DIR; }

/****************** nkhEnd: macro utils ******************/


/****************** nkhStart: global vars ******************/
time_t timerStart, timerEnd;
int timerCounter = 0;
double timerSec;
double timerFPS;
/****************** nkhEnd: global vars ******************/

void nkhMain(path inDir, path outDir, vector<path> frames);
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

int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);

    if (WITH_CUDA == 1) {
        cerr << "USING CUDA\n";
    } else {
        cerr << "NOT USING CUDA\n";
    }

    if (argc == 3)
    {
        path inDir(argv[1]), outDir(argv[2]);
        checkDir(inDir);
        checkDir(outDir);

        vector<path> inpVec;

        //TODO: handle non-regular files in dir, further handle non-images! I considered the inDir contains nothing than the frames.
        copy(directory_iterator(inDir), directory_iterator(), back_inserter(inpVec));
        sort(inpVec.begin(), inpVec.end());

        nkhMain(inDir, outDir, inpVec);

        return NORMAL_STATE;
    }
    else
    {
        cerr << errStr(error: INSUFFICIENT_ARGUMENTS\n);
        return INSUFFICIENT_ARGUMENTS;
    }
    //return a.exec();
}

void nkhMain(path inDir, path outDir, vector<path> frames)
{
    initFPSTimer();
    int fpsSum = 0, counter = 0 ;
    for (vector<path>::const_iterator it(frames.begin()), it_end(frames.end()); it != it_end; ++it)
    {
        fpsCalcStart();

        /*
        cv::Mat src = cv::imread(it->string(), cv::IMREAD_GRAYSCALE);
        cv::Mat dst;
        resize(src, dst, Size(640,480));
        cv::threshold(src, dst, 128.0, 255.0, cv::THRESH_BINARY);
        //cv::imshow("Result", dst);
        */

        cv::Mat src_host = cv::imread(it->string(), cv::IMREAD_GRAYSCALE), dst_host;
        cv::cuda::GpuMat dst, src;
        src.upload(src_host);
        cv::cuda::resize(src, dst, Size(640,480));
        cv::cuda::threshold(dst, src, 128.0, 255.0, cv::THRESH_BINARY);
        cv::Mat result_host(src);
        //cv::imshow("Result", result_host);


        /*
        Mat img = imread(it->string());
        GpuMat gpuImg ;
        gpuImg.upload(img);

        Mat dst = img.clone();
        gpuImg.download(dst);
        fpsCalcStart();
        imshow("Files", dst);
        */
        if (cvWaitKey(1) == 'q' )
            break;

        fpsCalcEnd();
        if(timerFPS !=INFINITY )
        {
            fpsSum += timerFPS;
            //cout << counter << " " << timerFPS << endl ;
        }
        counter++;
    }
    cout << (double) fpsSum/counter << endl;
}
