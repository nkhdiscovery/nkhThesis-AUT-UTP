#ifndef FRAMEOBJECTS_H
#define FRAMEOBJECTS_H

#include "PanelObject.h"
#include <vector>
#include <cstring>
#include <sstream>
using namespace std;

//TODO
// Object panels are considered to be rectangle

class FrameObjects{
private:
    vector<PanelObject> objects;
    int frameNumber;
public:
    FrameObjects()
    {
    }
    ~FrameObjects()
    {
//        delete &objects;
    }
    void add(PanelObject panelObj)
    {
        objects.push_back(panelObj);
    }
    void parse(string inString)
    {
        stringstream ss(inString);
        ss >> frameNumber;
        string tmpCat, tmpType, tmpName, tmpShape;
        while(ss >> tmpCat >> tmpType >> tmpName >> tmpShape)
        {
            if(tmpShape=="R")
            {
                int nums[7]={0};
                for (int i=0 ; i < 7 ; ss >> nums[i++]);
                objects.push_back(PanelObject(tmpCat, tmpType, tmpName,
                                                  cv::Rect(nums[0],nums[1], nums[2],nums[3]))
                                                  );
//                cout << tmpName << " : " <<
//                        nums[0] << ", " << nums[1] << ", " <<
//                        nums[2] << ", " << nums[3] << endl;
            }
        }
    }
    vector<PanelObject> getObjs()
    {
        return objects;
    }

    int getFrameNumber()
    {
        return frameNumber;
    }
};

#endif // FRAMEOBJECTS_H
