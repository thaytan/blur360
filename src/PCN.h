#ifndef __PCN__
#define __PCN__

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

#ifndef CLAMP
#define CLAMP(x, l, u)  ((x) < (l) ? (l) : ((x) > (u) ? (u) : (x)))
#endif

#ifndef M_PI
#define M_PI  3.14159265358979323846
#endif

#define EPS  1e-5

#define CYAN CV_RGB(0, 255, 255)
#define BLUE CV_RGB(0, 0, 255)
#define GREEN CV_RGB(0, 255, 0)
#define RED CV_RGB(255, 0, 0)
#define PURPLE CV_RGB(139, 0, 255)

struct Window
{
    int x, y, width, angle;
    float score;
    std::vector<cv::Point> points14;
    Window(int x_, int y_, int w_, int a_, float s_, std::vector<cv::Point> p14_)
        : x(x_), y(y_), width(w_), angle(a_), score(s_), points14(p14_)
    {}
};

cv::Point RotatePoint(float x, float y, float centerX, float centerY, float angle);
void DrawLine(cv::Mat img, std::vector<cv::Point> pointList);
void DrawFace(cv::Mat img, Window face);
void DrawPoints(cv::Mat img, Window face);
cv::Mat CropFace(cv::Mat img, Window face, int cropSize);

class PCN
{
public:
    PCN(std::string modelDetect, std::string net1, std::string net2, std::string net3,
        std::string modelTrack, std::string netTrack);
    /// detection
    void SetMinFaceSize(int minFace);
    void SetDetectionThresh(float thresh1, float thresh2, float thresh3);
    void SetImagePyramidScaleFactor(float factor);
    std::vector<Window> Detect(cv::Mat img);
    /// tracking
    void SetTrackingPeriod(int period);
    void SetTrackingThresh(float thresh);
    void SetVideoSmooth(bool smooth);
    std::vector<Window> DetectTrack(cv::Mat img);

private:
    void* impl_;
};

#endif
