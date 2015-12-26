#ifndef FRAMEOBJECTS_H
#define FRAMEOBJECTS_H

#include "PanelObject.h"
#include <vector>
#include <cstring>
#include <sstream>
using namespace std;

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
//        cout << " Frame num: " << frameNumber << endl;
        string tmpCat, tmpType, tmpName, tmpShape;
        while(ss >> tmpCat >> tmpType >> tmpName >> tmpShape)
        {
            if(tmpShape=="R")
            {
                int tmpNumbers[7]={0};
                for (int i=0 ; i < 7 ; ss >> tmpNumbers[i++]);
//                cout << tmpName << " : " <<
//                        tmpNumbers[0] << ", " << tmpNumbers[1] << ", " <<
//                        tmpNumbers[2] << ", " << tmpNumbers[3] << endl;
            }
        }
    }
    int getFrameNumber()
    {
        return frameNumber;
    }
};

#endif // FRAMEOBJECTS_H
