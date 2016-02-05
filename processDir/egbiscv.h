#ifndef EGBISCV_H
#define EGBISCV_H
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <vector>
using namespace cv;
using namespace std;
class nkhPoint{
public:
    int x, y;
    nkhPoint() {}
    nkhPoint(int _x, int _y)
    {
        x = _x;
        y = _y;
    }
    bool operator <(const nkhPoint& p) const{
        if(x != p.x)
            return x < p.x;
        else
            return y <= p.y;
    }
};

// Represent an edge between two pixels
class Edge {
public:
    int from;
    int to;
    nkhPoint pFrom, pTo;
    float weight;

    void setPoints(nkhPoint to, nkhPoint from)
    {
        pTo = to;
        pFrom = from;
    }

    bool operator <(const Edge& e) const {
        return weight < e.weight;
    }
};

// A point in the sets of points
class PointSetElement {
public:
    int p;
    int size;
//    vector<nkhPoint> coordVec;
//    cv::Mat shape;

    PointSetElement() { }
//    ~PointSetElement() {coordVec.clear();}
    PointSetElement(int p_) {
        p = p_;
        size = 1;
    }
//    void insertCoord(nkhPoint point)
//    {
//        coordVec.push_back(point);
//    }
};

// An object to manage set of points, who can be fusionned
class PointSet {
public:
    PointSet(int nb_elements_);
    ~PointSet();
    PointSet(){ mapping = NULL; }
    int nb_elements;

    // Return the main point of the point's set
    int getBasePoint(int p);

    // Join two sets of points, based on their main point
    void joinPoints(int p_a, int p_b);

    // Return the set size of a set (based on the main point)
    int size(unsigned int p) { return mapping[p].size; }

//    void insertCoord(nkhPoint point, int p)
//    {
//        mapping[p].insertCoord(point);
//    }

private:
    PointSetElement* mapping;

};

PointSet::PointSet(int nb_elements_) {
    nb_elements = nb_elements_;

    mapping = new PointSetElement[nb_elements];

    for ( int i = 0; i < nb_elements; i++) {
        mapping[i] = PointSetElement(i);
    }
}

PointSet::~PointSet() {
    delete [] mapping;
}

int PointSet::getBasePoint( int p) {

    int base_p = p;

    while (base_p != mapping[base_p].p) {
        base_p = mapping[base_p].p;
    }

    // Save mapping for faster acces later
    mapping[p].p = base_p;

    return base_p;
}

void PointSet::joinPoints(int p_a, int p_b) {

    // Always target smaller set, to avoid redirection in getBasePoint
    if (mapping[p_a].size < mapping[p_b].size)
        swap(p_a, p_b);

    mapping[p_b].p = p_a;
    mapping[p_a].size += mapping[p_b].size;
//    mapping[p_a].coordVec.insert(mapping[p_a].coordVec.end(), mapping[p_b].coordVec.begin(),mapping[p_b].coordVec.end());

    nb_elements--;
}


class nkhGraphSegmentationImpl{
public:
    nkhGraphSegmentationImpl() {
        sigma = 0.5;
        k = 300;
        min_size = 100;
        name_ = "GraphSegmentation";
    }

    ~nkhGraphSegmentationImpl() {
    };

    virtual void processImage(InputArray src, Mat &prun, OutputArray dst);

    virtual void setSigma(double sigma_) { if (sigma_ <= 0) { sigma_ = 0.001; } sigma = sigma_; }
    virtual double getSigma() { return sigma; }

    virtual void setK(float k_) { k = k_; }
    virtual float getK() { return k; }

    virtual void setMinSize(int min_size_) { min_size = min_size_; }
    virtual int getMinSize() { return min_size; }

    virtual void write(FileStorage& fs) const {
        fs << "name" << name_
           << "sigma" << sigma
           << "k" << k
           << "min_size" << (int)min_size;
    }

    virtual void read(const FileNode& fn) {
        CV_Assert( (String)fn["name"] == name_ );

        sigma = (double)fn["sigma"];
        k = (float)fn["k"];
        min_size = (int)(int)fn["min_size"];
    }

private:
    double sigma;
    float k;
    int min_size;
    String name_;

    // Pre-filter the image
    void filter(const Mat &img, Mat &img_filtered);

    // Build the graph between each pixels
    void buildGraph(Edge **edges, int &nb_edges, const Mat &costExtra, const Mat &img_filtered);

    // Segment the graph
    void segmentGraph(Edge * edges, const int &nb_edges, const Mat & img_filtered, PointSet **es);

    // Remove areas too small
    void filterSmallAreas(Edge *edges, const int &nb_edges, PointSet *es);

    // Map the segemented graph to a Mat with uniques, sequentials ids
    void finalMapping(PointSet *es, Mat &output);
};

void nkhGraphSegmentationImpl::filter(const Mat &img, Mat &img_filtered) {

    Mat img_converted;

    // Switch to float
    img.convertTo(img_converted, CV_32F);

    // Apply gaussian filter
    GaussianBlur(img_converted, img_filtered, Size(0, 0), sigma, sigma);
}

void nkhGraphSegmentationImpl::buildGraph(Edge **edges, int &nb_edges, const Mat &costExtra, const Mat &img_filtered) {

    *edges = new Edge[img_filtered.rows * img_filtered.cols * 4];

    srand(time(NULL));
    nb_edges = 0;
    int counter = 0;
    int nb_channels = img_filtered.channels();
    int nb_cost_channels = costExtra.channels();

    for (int i = 0; i < (int)img_filtered.rows; i++) {
        const float* p = img_filtered.ptr<float>(i);
        const uchar* ptrCost = costExtra.ptr<uchar>(i);

        for (int j = 0; j < (int)img_filtered.cols; j++) {

            //Take the right, left, top and down pixel
            for (int delta = -1; delta <= 1; delta += 2) {
                for (int delta_j = 0, delta_i = 1; delta_j <= 1; delta_j++ || delta_i--) {

                    int i2 = i + delta * delta_i;
                    int j2 = j + delta * delta_j;

                    if (i2 >= 0 && i2 < img_filtered.rows && j2 >= 0 && j2 < img_filtered.cols) {

                        const float* p2 = img_filtered.ptr<float>(i2);
                        const uchar* ptr2Cost = costExtra.ptr<uchar>(i2);

                        float tmp_total = 0;

                        for ( int channel = 0; channel < nb_channels; channel++) {
                            tmp_total += pow(p[j * nb_channels + channel] - p2[j2 * nb_channels + channel], 2);
                        }

                        float diff = 0;
                        diff = sqrt(tmp_total);

                        for ( int channel = 0; channel < nb_cost_channels; channel++) {
                            tmp_total += ( (ptrCost[j * nb_cost_channels + channel]
                                    == 255) && (ptr2Cost[j2 * nb_cost_channels + channel] == 255) && tmp_total<=5) ? 0.1*tmp_total : 2000.0*tmp_total;
//                            cout << (int)ptr2Cost[j2 * nb_cost_channels + channel] << " " ;
                        }

                        //pruning matrix condition check
//                        cout << (int)ptrPrun[j] <<" , " <<(int)ptr2Prun[j2] << endl ;
//                        if( rand() % 7 && ptrPrun[j] >0 && ptr2Prun[j2] > 0)
//                        if( ptr2Prun[j] | ptr2Prun[j2])
//                        {
//                            continue;
//                        }

//                        if(rand()%50 < diff) continue;

                        (*edges)[nb_edges].weight = diff;
                        (*edges)[nb_edges].from = i * img_filtered.cols +  j;
                        (*edges)[nb_edges].to = i2 * img_filtered.cols + j2;
//                        (*edges)[nb_edges].setPoints(nkhPoint(i,j), nkhPoint(i2,j2));
                        nb_edges++;
                    }

                    counter++;
                }
            }
        }
    }
}

void nkhGraphSegmentationImpl::segmentGraph(Edge *edges, const int &nb_edges, const Mat &img_filtered, PointSet **es) {

    int total_points = ( int)(img_filtered.rows * img_filtered.cols);

    // Sort edges
    std::sort(edges, edges + nb_edges);

    // Create a set with all point (by default mapped to themselfs)
    *es = new PointSet(img_filtered.cols * img_filtered.rows);

    // Thresholds
    float* thresholds = new float[total_points];

    for (int i = 0; i < total_points; i++)
        thresholds[i] = k;

    for ( int i = 0; i < nb_edges; i++) {

        int p_a = (*es)->getBasePoint(edges[i].from);
        int p_b = (*es)->getBasePoint(edges[i].to);
//        (*es)->insertCoord(edges[i].pFrom, p_a);
//        (*es)->insertCoord(edges[i].pTo, p_b);


        if (p_a != p_b) {
            if (edges[i].weight <= thresholds[p_a] && edges[i].weight <= thresholds[p_b]) {
                (*es)->joinPoints(p_a, p_b);
                p_a = (*es)->getBasePoint(p_a);
                thresholds[p_a] = edges[i].weight + k / (*es)->size(p_a);

                edges[i].weight = 0;
            }
        }
    }

    delete [] thresholds;
}

void nkhGraphSegmentationImpl::filterSmallAreas(Edge *edges, const int &nb_edges, PointSet *es) {

    for ( int i = 0; i < nb_edges; i++) {

        if (edges[i].weight > 0) {

            int p_a = es->getBasePoint(edges[i].from);
            int p_b = es->getBasePoint(edges[i].to);

            if (p_a != p_b && (es->size(p_a) < min_size || es->size(p_b) < min_size)) {
                es->joinPoints(p_a, p_b);
            }
        }
    }

}

void nkhGraphSegmentationImpl::finalMapping(PointSet *es, Mat &output) {

    int maximum_size = ( int)(output.rows * output.cols);

    int last_id = 0;
    int * mapped_id = new int[maximum_size];

    for ( int i = 0; i < maximum_size; i++)
        mapped_id[i] = -1;

    int rows = output.rows;
    int cols = output.cols;

    if (output.isContinuous()) {
        cols *= rows;
        rows = 1;
    }

    for (int i = 0; i < rows; i++) {

        int* p = output.ptr<int>(i);

        for (int j = 0; j < cols; j++) {

            int point = es->getBasePoint(i * cols + j);

            if (mapped_id[point] == -1) {
                mapped_id[point] = last_id;
                last_id++;
            }

            p[j] = mapped_id[point];
        }
    }

    free(mapped_id);
}

void nkhGraphSegmentationImpl::processImage(InputArray src, cv::Mat& prun, OutputArray dst) {

    Mat img = src.getMat();

    dst.create(img.rows, img.cols, CV_32SC1);
    Mat output = dst.getMat();
    output.setTo(0);

    // Filter graph
    Mat img_filtered;
    filter(img, img_filtered);

    // Build graph
    Edge *edges;
    int nb_edges;

    buildGraph(&edges, nb_edges, prun, img_filtered);

    // Segment graph
    PointSet *es;

    segmentGraph(edges, nb_edges, img_filtered, &es);

    // Remove small areas
    filterSmallAreas(edges, nb_edges, es);

    // Map to final output
    finalMapping(es, output);

    free(edges);
    delete es;

}

Ptr<nkhGraphSegmentationImpl> createGraphSegmentation(double sigma, float k, int min_size) {

    Ptr<nkhGraphSegmentationImpl> graphseg = makePtr<nkhGraphSegmentationImpl>();

    graphseg->setSigma(sigma);
    graphseg->setK(k);
    graphseg->setMinSize(min_size);

    return graphseg;
}

int egbisVisualise(cv::Mat &egbisImage, cv::Mat& visualEgbis)
{
    double min, max;
    cv::minMaxLoc(egbisImage, &min, &max);
    int nb_segs = (int)max + 1;
    cv::Mat channs[3];
    convertScaleAbs(egbisImage, channs[0], 255/max);
    channs[0].convertTo(channs[0], CV_8UC1);
    channs[1] = cv::Mat::zeros(egbisImage.rows, egbisImage.cols, CV_8UC1) + 200;
    channs[2] = cv::Mat::zeros(egbisImage.rows, egbisImage.cols, CV_8UC1) + 200;
    visualEgbis = cv::Mat::zeros(egbisImage.rows, egbisImage.cols, CV_8UC3);
    cv::merge(channs, 3, visualEgbis);
    cvtColor(visualEgbis, visualEgbis, COLOR_HSV2RGB);
    std::cout << nb_segs << " segments" << std::endl;
    return nb_segs;
//    visualEgbis = cv::Mat::zeros(egbisImage.rows, egbisImage.cols, CV_8UC3);
//    uint* p;
//    uchar* p2;
//    for (int i = 0; i < egbisImage.rows; i++) {
//        p = egbisImage.ptr<uint>(i);
//        p2 = visualEgbis.ptr<uchar>(i);
//        for (int j = 0; j < egbisImage.cols; j++) {
//            cv::Scalar color = color_mapping(p[j]);
//            p2[j*3] = color[0] ;
//            p2[j*3 + 1] = color[1];
//            p2[j*3 + 2] = color[2];
//        }
//    }
}

#endif // EGBISCV_H
