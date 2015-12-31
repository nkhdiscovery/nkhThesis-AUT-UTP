#ifndef PANELOBJECT_H
#define PANELOBJECT_H

#include <opencv2/core.hpp>
#include <cstring>

#define PANEL_CATEGORY "Panel"
#define PANEL_TYPE_ABOVE "Above"
#define PANEL_TYPE_SIDE "Side"

#define TEXT_CATEGORY "Text"
#define TEXT_TYPE_PERSIAN "Persian"
#define TEXT_TYPE_ENGLISH "English"

//TODO
// Object panels are considered to be rectangle

class PanelObject{
protected:
    string category;
    string type;
    cv::Rect border;
    string name;
public:
    PanelObject(string _objectCat, string _objectType, string _name, cv::Rect _border){
        category = _objectCat;
        type = _objectType;
        border = _border;
        name = _name;
    }
    string getCategory(){return category;}
    string getType(){return type;}
    string getName(){return name;}
    cv::Rect getBorder(){return border;}


};

#endif // PANELOBJECT_H
