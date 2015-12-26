#ifndef PANELOBJECT_H
#define PANELOBJECT_H

#include <opencv2/core.hpp>
#include <cstring>

enum{
    PANEL_SIDE,
    PANEL_ABOVE,
    PERSIAN_TEXT,
    ENGLISH_TEXT
};
class PanelObject{
private:
    int objectType;
    cv::Rect border;
    string name;
public:
    PanelObject(int _objectType, string _name, cv::Rect _border){
        objectType = _objectType;
        border = _border;
        name = _name;
    }
};

#endif // PANELOBJECT_H
