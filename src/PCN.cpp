#include "PCN.h"
#include "PCN_API.h"

struct Window2
{
    int x, y, w, h;
    float angle, scale, conf;
    std::vector<cv::Point> points14;
    Window2(int x_, int y_, int w_, int h_, float a_, float s_, float c_)
        : x(x_), y(y_), w(w_), h(h_), angle(a_), scale(s_), conf(c_)
    {}
};

class Impl
{
public:
    void LoadModel(std::string modelDetect, std::string net1, std::string net2, std::string net3,
                   std::string modelTrack, std::string netTrack);
    cv::Mat ResizeImg(cv::Mat img, float scale);
    static bool CompareWin(const Window2 &w1, const Window2 &w2);
    bool Legal(int x, int y, cv::Mat img);
    bool Inside(int x, int y, Window2 rect);
    int SmoothAngle(int a, int b);
    std::vector<Window2> SmoothWindow(std::vector<Window2> winList);
    float IoU(Window2 &w1, Window2 &w2);
    std::vector<Window2> NMS(std::vector<Window2> &winList, bool local, float threshold);
    std::vector<Window2> DeleteFP(std::vector<Window2> &winList);
    cv::Mat PreProcessImg(cv::Mat img);
    cv::Mat PreProcessImg(cv::Mat img,  int dim);
    cv::Mat PadImg(cv::Mat img);
    std::vector<Window> TransWindow(cv::Mat img, cv::Mat imgPad, std::vector<Window2> &winList);
    std::vector<Window2> Stage1(cv::Mat img, cv::Mat imgPad, cv::dnn::Net &net, float thres);
    std::vector<Window2> Stage2(cv::Mat img, cv::Mat img180,
                                cv::dnn::Net &net, float thres, int dim, std::vector<Window2> &winList);
    std::vector<Window2> Stage3(cv::Mat img, cv::Mat img180, cv::Mat img90, cv::Mat imgNeg90,
                                cv::dnn::Net &net, float thres, int dim, std::vector<Window2> &winList);
    std::vector<Window2> Detect(cv::Mat img, cv::Mat imgPad);
    std::vector<Window2> Track(cv::Mat img, cv::dnn::Net &net,
                               float thres, int dim, std::vector<Window2> &winList);
public:
    cv::dnn::Net net_[4];
    int minFace_;
    float scale_;
    int stride_;
    float classThreshold_[3];
    float nmsThreshold_[3];
    float angleRange_;
    bool stable_;
    int period_;
    float trackThreshold_;
    float augScale_;
    cv::Scalar mean_;
};

PCN::PCN(std::string modelDetect, std::string net1, std::string net2, std::string net3,
         std::string modelTrack, std::string netTrack) : impl_(new Impl())
{
    Impl *p = (Impl *)impl_;
    p->LoadModel(modelDetect, net1, net2, net3, modelTrack, netTrack);
}

void PCN::SetVideoSmooth(bool stable)
{
    Impl *p = (Impl *)impl_;
    p->stable_ = stable;
}

void PCN::SetMinFaceSize(int minFace)
{
    Impl *p = (Impl *)impl_;
    p->minFace_ = minFace > 20 ? minFace : 20;
    p->minFace_ *= 1.4;
}

void PCN::SetDetectionThresh(float thresh1, float thresh2, float thresh3)
{
    Impl *p = (Impl *)impl_;
    p->classThreshold_[0] = thresh1;
    p->classThreshold_[1] = thresh2;
    p->classThreshold_[2] = thresh3;
    p->nmsThreshold_[0] = 0.8;
    p->nmsThreshold_[1] = 0.8;
    p->nmsThreshold_[2] = 0.3;
    p->stride_ = 8;
    p->angleRange_ = 45;
    p->augScale_ = 0.15;
    p->mean_ = cv::Scalar(104, 117, 123);
}

void PCN::SetImagePyramidScaleFactor(float factor)
{
    Impl *p = (Impl *)impl_;
    p->scale_ = factor;
}

void PCN::SetTrackingPeriod(int period)
{
    Impl *p = (Impl *)impl_;
    p->period_ = period;
}

void PCN::SetTrackingThresh(float thres)
{
    Impl *p = (Impl *)impl_;
    p->trackThreshold_ = thres;
}

std::vector<Window> PCN::Detect(cv::Mat img)
{
    Impl *p = (Impl *)impl_;
    cv::Mat imgPad = p->PadImg(img);
    std::vector<Window2> winList = p->Detect(img, imgPad);
    std::vector<Window2> pointsList = p->Track(imgPad, p->net_[3], -1, 96, winList);
    for (size_t i = 0; i < winList.size(); i++)
    {
        winList[i].points14 = pointsList[i].points14;
    }

    if (p->stable_)
    {
        winList = p->SmoothWindow(winList);
    }
    return p->TransWindow(img, imgPad, winList);
}

std::vector<Window> PCN::DetectTrack(cv::Mat img)
{
    Impl *p = (Impl *)impl_;
    cv::Mat imgPad = p->PadImg(img);

    static int detectFlag = p->period_;
    static std::vector<Window2> preList;
    std::vector<Window2> winList = preList;

    if (detectFlag == p->period_)
    {
        std::vector<Window2> tmpList = p->Detect(img, imgPad);
        for (size_t i = 0; i < tmpList.size(); i++)
        {
            winList.push_back(tmpList[i]);
        }
    }
    winList = p->NMS(winList, false, p->nmsThreshold_[2]);
    winList = p->Track(imgPad, p->net_[3], p->trackThreshold_, 96, winList);
    winList = p->NMS(winList, false, p->nmsThreshold_[2]);
    winList = p->DeleteFP(winList);
    if (p->stable_)
    {
        winList = p->SmoothWindow(winList);
    }
    preList = winList;
    detectFlag--;
    if (detectFlag == 0)
        detectFlag = p->period_;
    return p->TransWindow(img, imgPad, winList);
}

void Impl::LoadModel(std::string modelDetect, std::string net1, std::string net2, std::string net3,
                     std::string modelTrack, std::string netTrack)
{
    net_[0] = cv::dnn::readNetFromCaffe(net1.c_str(), modelDetect.c_str());
    net_[1] = cv::dnn::readNetFromCaffe(net2.c_str(), modelDetect.c_str());
    net_[2] = cv::dnn::readNetFromCaffe(net3.c_str(), modelDetect.c_str());
    net_[3] = cv::dnn::readNetFromCaffe(netTrack.c_str(), modelTrack.c_str());
}

cv::Mat Impl::PreProcessImg(cv::Mat img)
{
    cv::Mat mean(img.size(), CV_32FC3, mean_);
    cv::Mat imgF;
    img.convertTo(imgF, CV_32FC3);
    return imgF - mean;
}

cv::Mat Impl::PreProcessImg(cv::Mat img, int dim)
{
    cv::Mat imgNew;
    cv::resize(img, imgNew, cv::Size(dim, dim));
    cv::Mat mean(imgNew.size(), CV_32FC3, mean_);
    cv::Mat imgF;
    imgNew.convertTo(imgF, CV_32FC3);
    return imgF - mean;
}

cv::Mat Impl::ResizeImg(cv::Mat img, float scale)
{
    cv::Mat ret;
    cv::resize(img, ret, cv::Size(int(img.cols / scale), int(img.rows / scale)));
    return ret;
}

bool Impl::CompareWin(const Window2 &w1, const Window2 &w2)
{
    return w1.conf > w2.conf;
}

bool Impl::Legal(int x, int y, cv::Mat img)
{
    if (x >= 0 && x < img.cols && y >= 0 && y < img.rows)
        return true;
    else
        return false;
}

bool Impl::Inside(int x, int y, Window2 rect)
{
    if (x >= rect.x && y >= rect.y && x < rect.x + rect.w && y < rect.y + rect.h)
        return true;
    else
        return false;
}

int Impl::SmoothAngle(int a, int b)
{
    if (a > b)
        std::swap(a, b);
    int diff = (b - a) % 360;
    if (diff < 180)
        return a + diff / 2;
    else
        return b + (360 - diff) / 2;
}

float Impl::IoU(Window2 &w1, Window2 &w2)
{
    float xOverlap = std::max(0, std::min(w1.x + w1.w - 1, w2.x + w2.w - 1) - std::max(w1.x, w2.x) + 1);
    float yOverlap = std::max(0, std::min(w1.y + w1.h - 1, w2.y + w2.h - 1) - std::max(w1.y, w2.y) + 1);
    float intersection = xOverlap * yOverlap;
    float unio = w1.w * w1.h + w2.w * w2.h - intersection;
    return float(intersection) / unio;
}

std::vector<Window2> Impl::NMS(std::vector<Window2> &winList, bool local, float threshold)
{
    if (winList.size() == 0)
        return winList;
    std::sort(winList.begin(), winList.end(), CompareWin);

    std::vector<bool> flag(winList.size(), false);

    for (size_t i = 0; i < winList.size(); i++)
    {
        if (flag[i])
            continue;

        for (size_t j = i + 1; j < winList.size(); j++)
        {
            if (local && abs(winList[i].scale - winList[j].scale) > EPS)
                continue;
            if (IoU(winList[i], winList[j]) > threshold)
                flag[j] = true;
        }
    }
    std::vector<Window2> ret;
    for (size_t i = 0; i < winList.size(); i++)
    {
        if (!flag[i]) ret.push_back(winList[i]);
    }
    return ret;
}

/// to delete some false positives
std::vector<Window2> Impl::DeleteFP(std::vector<Window2> &winList)
{
    if (winList.size() == 0)
        return winList;
    std::sort(winList.begin(), winList.end(), CompareWin);
    bool flag[winList.size()];
    memset(flag, 0, winList.size());
    for (size_t i = 0; i < winList.size(); i++)
    {
        if (flag[i])
            continue;
        for (size_t j = i + 1; j < winList.size(); j++)
        {
            if (Inside(winList[j].x, winList[j].y, winList[i]) && Inside(winList[j].x + winList[j].w - 1, winList[j].y + winList[j].h - 1, winList[i]))
                flag[j] = 1;
        }
    }
    std::vector<Window2> ret;
    for (size_t i = 0; i < winList.size(); i++)
    {
        if (!flag[i]) ret.push_back(winList[i]);
    }
    return ret;
}

/// to detect faces on the boundary
cv::Mat Impl::PadImg(cv::Mat img)
{
    int row = std::min(int(img.rows * 0.2), 100);
    int col = std::min(int(img.cols * 0.2), 100);
    cv::Mat ret;
    cv::copyMakeBorder(img, ret, row, row, col, col, cv::BORDER_CONSTANT, mean_);
    return ret;
}

std::vector<Window2> Impl::Stage1(cv::Mat img, cv::Mat imgPad, cv::dnn::Net &net, float thres)
{
    std::vector<cv::String> outputBlobNames = { "bbox_reg_1", "cls_prob", "rotate_cls_prob" };

    int row = (imgPad.rows - img.rows) / 2;
    int col = (imgPad.cols - img.cols) / 2;
    std::vector<Window2> winList;
    int netSize = 24;
    float curScale;
    curScale = minFace_ / float(netSize);
    cv::Mat imgResized = ResizeImg(img, curScale);
    while (std::min(imgResized.rows, imgResized.cols) >= netSize)
    {
        cv::Mat preProcessed = PreProcessImg(imgResized);
        cv::Mat inputBlob = cv::dnn::blobFromImage(preProcessed, 1.0, cv::Size(), cv::Scalar(), false, false);
        std::vector<cv::Mat> outputBlobs;

        net.setInput(inputBlob);
        net.forward(outputBlobs, outputBlobNames);

        cv::Mat regression[3] = {
            cv::Mat(outputBlobs[0].size[2], outputBlobs[0].size[3], CV_32F, outputBlobs[0].ptr<float>(0,0)),
            cv::Mat(outputBlobs[0].size[2], outputBlobs[0].size[3], CV_32F, outputBlobs[0].ptr<float>(0,1)),
            cv::Mat(outputBlobs[0].size[2], outputBlobs[0].size[3], CV_32F, outputBlobs[0].ptr<float>(0,2)),
        };

        cv::Mat prob        = cv::Mat(outputBlobs[1].size[2], outputBlobs[1].size[3], CV_32F, outputBlobs[1].ptr<float>(0,1));
        cv::Mat rotateProbs = cv::Mat(outputBlobs[2].size[2], outputBlobs[2].size[3], CV_32F, outputBlobs[2].ptr<float>(0,1));

        float w = netSize * curScale;
        for (int i = 0; i < prob.rows; i++)
        {
            for (int j = 0; j < prob.cols; j++)
            {
                float faceProbability = prob.at<float>(i, j);

                if (faceProbability > thres)
                {
                    float sn = regression[0].at<float>(i, j);
                    float xn = regression[1].at<float>(i, j);
                    float yn = regression[2].at<float>(i, j);

                    int rx = j * curScale * stride_ - 0.5 * sn * w + sn * xn * w + 0.5 * w + col;
                    int ry = i * curScale * stride_ - 0.5 * sn * w + sn * yn * w + 0.5 * w + row;
                    int rw = w * sn;

                    if (Legal(rx, ry, imgPad) && Legal(rx + rw - 1, ry + rw - 1, imgPad))
                    {
                        if (rotateProbs.at<float>(i, j) > 0.5)
                            winList.push_back(Window2(rx, ry, rw, rw, 0, curScale, faceProbability));
                        else
                            winList.push_back(Window2(rx, ry, rw, rw, 180, curScale, faceProbability));
                    }
                }
            }
        }
        imgResized = ResizeImg(imgResized, scale_);
        curScale = float(img.rows) / imgResized.rows;
    }
    return winList;
}

std::vector<Window2> Impl::Stage2(cv::Mat img, cv::Mat img180, cv::dnn::Net &net, float thres, int dim, std::vector<Window2> &winList)
{
    if (winList.size() == 0)
        return winList;
    std::vector<cv::Mat> dataList;
    dataList.reserve(winList.size());

    int height = img.rows;
    for (size_t i = 0; i < winList.size(); i++)
    {
        if (abs(winList[i].angle) < EPS)
            dataList.push_back(PreProcessImg(img(cv::Rect(winList[i].x, winList[i].y, winList[i].w, winList[i].h)), dim));
        else
        {
            int y2 = winList[i].y + winList[i].h - 1;
            dataList.push_back(PreProcessImg(img180(cv::Rect(winList[i].x, height - 1 - y2, winList[i].w, winList[i].h)), dim));
        }
    }

    std::vector<cv::String> outputBlobNames = { "bbox_reg_2", "cls_prob", "rotate_cls_prob" };
    std::vector<cv::Mat> outputBlobs;
    std::vector<Window2> ret;

#if 0
    /* FIXME: Figure out how the reports from multiple images work so all images can be submitted at once */
    cv::Mat inputBlob = cv::dnn::blobFromImages(dataList, 1.0, cv::Size(), cv::Scalar(), false, false);
    net.setInput(inputBlob);
    net.forward(outputBlobs, outputBlobNames);

    for (size_t b = 0; b < outputBlobs.size(); b++) {
        std::cout << "Stage 2 output blob " << b << " is " << outputBlobs[b].dims << " dimensional" << std::endl;
        for (int d = 0; d < outputBlobs[b].dims; d++) {
            std::cout << "Dim " << d << " = " << outputBlobs[b].size[d] << std::endl;
        }
    }

    cv::Mat regression  = cv::Mat(outputBlobs[0].size[1], outputBlobs[0].size[0], CV_32F, outputBlobs[0].ptr<float>(0,0));
    cv::Mat prob        = cv::Mat(outputBlobs[1].size[1], outputBlobs[1].size[0], CV_32F, outputBlobs[1].ptr<float>(0,0));
    cv::Mat rotateProbs = cv::Mat(outputBlobs[2].size[1], outputBlobs[2].size[0], CV_32F, outputBlobs[2].ptr<float>(0,0));

    for (size_t i = 0; i < winList.size(); i++)
    {

        float score = prob.at<float>(1, i);

        if (score > thres)
        {
            float sn = regression.at<float>(i, 0);
            float xn = regression.at<float>(i, 1);
            float yn = regression.at<float>(i, 2);

            std::cout << "Candidate " << i << " score " << score << " [sn, xn, yn] = [" <<
                sn << ", " << xn << ", " << yn << "]" << std::endl;

            int cropX = winList[i].x;
            int cropY = winList[i].y;
            int cropW = winList[i].w;
            if (abs(winList[i].angle)  > EPS)
                cropY = height - 1 - (cropY + cropW - 1);
            int w = sn * cropW;
            int x = cropX  - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW;
            int y = cropY  - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW;
            float maxRotateScore = 0;
            int maxRotateIndex = 0;
            for (int j = 0; j < 3; j++)
            {
                float rotateScore = rotateProbs.at<float>(j, i);

                std::cout << "Candidate " << i << " rotate score " << j << " = " << rotateScore << std::endl;

                if (rotateScore > maxRotateScore)
                {
                    maxRotateScore = rotateScore;
                    maxRotateIndex = j;
                }
            }
            if (Legal(x, y, img) && Legal(x + w - 1, y + w - 1, img))
            {
                float angle = 0;
                if (abs(winList[i].angle)  < EPS)
                {
                    if (maxRotateIndex == 0)
                        angle = 90;
                    else if (maxRotateIndex == 1)
                        angle = 0;
                    else
                        angle = -90;
                    ret.push_back(Window2(x, y, w, w, angle, winList[i].scale, score));
                }
                else
                {
                    if (maxRotateIndex == 0)
                        angle = 90;
                    else if (maxRotateIndex == 1)
                        angle = 180;
                    else
                        angle = -90;
                    ret.push_back(Window2(x, height - 1 -  (y + w - 1), w, w, angle, winList[i].scale, score));
                }
            }
        }
    }
#else
    for (size_t i = 0; i < winList.size(); i++)
    {
        cv::Mat inputBlob = cv::dnn::blobFromImage(dataList[i], 1.0, cv::Size(), cv::Scalar(), false, false);
        net.setInput(inputBlob);
        net.forward(outputBlobs, outputBlobNames);

        cv::Mat regression  = cv::Mat(outputBlobs[0].size[1], outputBlobs[0].size[0], CV_32F, outputBlobs[0].ptr<float>(0,0));
        cv::Mat prob        = cv::Mat(outputBlobs[1].size[1], outputBlobs[1].size[0], CV_32F, outputBlobs[1].ptr<float>(0,0));
        cv::Mat rotateProbs = cv::Mat(outputBlobs[2].size[1], outputBlobs[2].size[0], CV_32F, outputBlobs[2].ptr<float>(0,0));

        float score = prob.at<float>(1, 0);

        if (score > thres)
        {
            float sn = regression.at<float>(0, 0);
            float xn = regression.at<float>(1, 0);
            float yn = regression.at<float>(2, 0);

            int cropX = winList[i].x;
            int cropY = winList[i].y;
            int cropW = winList[i].w;
            if (abs(winList[i].angle)  > EPS)
                cropY = height - 1 - (cropY + cropW - 1);
            int w = sn * cropW;
            int x = cropX  - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW;
            int y = cropY  - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW;
            float maxRotateScore = 0;
            int maxRotateIndex = 0;
            for (int j = 0; j < 3; j++)
            {
                float rotateScore = rotateProbs.at<float>(j, 0);

                if (rotateScore > maxRotateScore)
                {
                    maxRotateScore = rotateScore;
                    maxRotateIndex = j;
                }
            }
            if (Legal(x, y, img) && Legal(x + w - 1, y + w - 1, img))
            {
                float angle = 0;
                if (abs(winList[i].angle)  < EPS)
                {
                    if (maxRotateIndex == 0)
                        angle = 90;
                    else if (maxRotateIndex == 1)
                        angle = 0;
                    else
                        angle = -90;
                    ret.push_back(Window2(x, y, w, w, angle, winList[i].scale, score));
                }
                else
                {
                    if (maxRotateIndex == 0)
                        angle = 90;
                    else if (maxRotateIndex == 1)
                        angle = 180;
                    else
                        angle = -90;
                    ret.push_back(Window2(x, height - 1 -  (y + w - 1), w, w, angle, winList[i].scale, score));
                }
            }
        }
    }
#endif
    return ret;
}

std::vector<Window2> Impl::Stage3(cv::Mat img, cv::Mat img180, cv::Mat img90, cv::Mat imgNeg90, cv::dnn::Net &net, float thres, int dim, std::vector<Window2> &winList)
{
    if (winList.size() == 0)
        return winList;
    std::vector<cv::Mat> dataList;
    int height = img.rows;
    int width = img.cols;
    for (size_t i = 0; i < winList.size(); i++)
    {
        if (abs(winList[i].angle) < EPS)
            dataList.push_back(PreProcessImg(img(cv::Rect(winList[i].x, winList[i].y, winList[i].w, winList[i].h)), dim));
        else if (abs(winList[i].angle - 90) < EPS)
        {
            dataList.push_back(PreProcessImg(img90(cv::Rect(winList[i].y, winList[i].x, winList[i].h, winList[i].w)), dim));
        }
        else if (abs(winList[i].angle + 90) < EPS)
        {
            int x = winList[i].y;
            int y = width - 1 - (winList[i].x + winList[i].w - 1);
            dataList.push_back(PreProcessImg(imgNeg90(cv::Rect(x, y, winList[i].w, winList[i].h)), dim));
        }
        else
        {
            int y2 = winList[i].y + winList[i].h - 1;
            dataList.push_back(PreProcessImg(img180(cv::Rect(winList[i].x, height - 1 - y2, winList[i].w, winList[i].h)), dim));
        }
    }

    std::vector<cv::String> outputBlobNames = { "bbox_reg_3", "cls_prob", "rotate_reg_3" };
    std::vector<cv::Mat> outputBlobs;
    std::vector<Window2> ret;

    for (size_t i = 0; i < winList.size(); i++)
    {
        cv::Mat inputBlob = cv::dnn::blobFromImage(dataList[i], 1.0, cv::Size(), cv::Scalar(), false, false);
        net.setInput(inputBlob);
        net.forward(outputBlobs, outputBlobNames);


        cv::Mat regression  = cv::Mat(outputBlobs[0].size[1], outputBlobs[0].size[0], CV_32F, outputBlobs[0].ptr<float>(0,0));
        cv::Mat prob        = cv::Mat(outputBlobs[1].size[1], outputBlobs[1].size[0], CV_32F, outputBlobs[1].ptr<float>(0,0));
        cv::Mat rotateProbs = cv::Mat(outputBlobs[2].size[1], outputBlobs[2].size[0], CV_32F, outputBlobs[2].ptr<float>(0,0));

        float score = prob.at<float>(1, 0);

        if (score > thres)
        {
            float sn = regression.at<float>(0, 0);
            float xn = regression.at<float>(1, 0);
            float yn = regression.at<float>(2, 0);

            int cropX = winList[i].x;
            int cropY = winList[i].y;
            int cropW = winList[i].w;
            cv::Mat imgTmp = img;
            if (abs(winList[i].angle - 180)  < EPS)
            {
                cropY = height - 1 - (cropY + cropW - 1);
                imgTmp = img180;
            }
            else if (abs(winList[i].angle - 90)  < EPS)
            {
                std::swap(cropX, cropY);
                imgTmp = img90;
            }
            else if (abs(winList[i].angle + 90)  < EPS)
            {
                cropX = winList[i].y;
                cropY = width - 1 - (winList[i].x + winList[i].w - 1);
                imgTmp = imgNeg90;
            }

            int w = sn * cropW;
            int x = cropX  - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW;
            int y = cropY  - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW;
            float angle = angleRange_ * rotateProbs.at<float>(0, 0);

            if (Legal(x, y, imgTmp) && Legal(x + w - 1, y + w - 1, imgTmp))
            {
                if (abs(winList[i].angle)  < EPS)
                    ret.push_back(Window2(x, y, w, w, angle, winList[i].scale, score));
                else if (abs(winList[i].angle - 180)  < EPS)
                {
                    ret.push_back(Window2(x, height - 1 -  (y + w - 1), w, w, 180 - angle, winList[i].scale, score));
                }
                else if (abs(winList[i].angle - 90)  < EPS)
                {
                    ret.push_back(Window2(y, x, w, w, 90 - angle, winList[i].scale, score));
                }
                else
                {
                    ret.push_back(Window2(width - y - w, x, w, w, -90 + angle, winList[i].scale, score));
                }
            }
        }
    }

    return ret;
}

std::vector<Window> Impl::TransWindow(cv::Mat img, cv::Mat imgPad, std::vector<Window2> &winList)
{
    int row = (imgPad.rows - img.rows) / 2;
    int col = (imgPad.cols - img.cols) / 2;

    std::vector<Window> ret;
    for(size_t i = 0; i < winList.size(); i++)
    {
        if (winList[i].w > 0 && winList[i].h > 0)
        {
            for (size_t j = 0; j < winList[i].points14.size(); j++)
            {
                winList[i].points14[j].x -= col;
                winList[i].points14[j].y -= row;
            }
            ret.push_back(Window(winList[i].x - col, winList[i].y - row, winList[i].w, winList[i].angle, winList[i].conf, winList[i].points14));
        }
    }
    return ret;
}

std::vector<Window2> Impl::SmoothWindow(std::vector<Window2> winList)
{
    static std::vector<Window2> preList;
    for (size_t i = 0; i < winList.size(); i++)
    {
        for (size_t j = 0; j < preList.size(); j++)
        {
            if (IoU(winList[i], preList[j]) > 0.9)
            {
                winList[i].conf = (winList[i].conf + preList[j].conf) / 2;
                winList[i].x = preList[j].x;
                winList[i].y = preList[j].y;
                winList[i].w = preList[j].w;
                winList[i].h = preList[j].h;
                winList[i].angle = preList[j].angle;
                for (size_t k = 0; k < preList[j].points14.size(); k++)
                {
                    winList[i].points14[k].x = (4 * winList[i].points14[k].x + 6 * preList[j].points14[k].x) / 10.0;
                    winList[i].points14[k].y = (4 * winList[i].points14[k].y + 6 * preList[j].points14[k].y) / 10.0;
                }
            }
            else if (IoU(winList[i], preList[j]) > 0.6)
            {
                winList[i].conf = (winList[i].conf + preList[j].conf) / 2;
                winList[i].x = (winList[i].x + preList[j].x) / 2;
                winList[i].y = (winList[i].y + preList[j].y) / 2;
                winList[i].w = (winList[i].w + preList[j].w) / 2;
                winList[i].h = (winList[i].h + preList[j].h) / 2;
                winList[i].angle = SmoothAngle(winList[i].angle, preList[j].angle);
                for (size_t k = 0; k < preList[j].points14.size(); k++)
                {
                    winList[i].points14[k].x = (7 * winList[i].points14[k].x + 3 * preList[j].points14[k].x) / 10.0;
                    winList[i].points14[k].y = (7 * winList[i].points14[k].y + 3 * preList[j].points14[k].y) / 10.0;
                }
            }
        }
    }
    preList = winList;
    return winList;
}

std::vector<Window2> Impl::Detect(cv::Mat img, cv::Mat imgPad)
{
    cv::Mat img180, img90, imgNeg90;
    cv::flip(imgPad, img180, 0);
    cv::transpose(imgPad, img90);
    cv::flip(img90, imgNeg90, 0);

    std::vector<Window2> winList = Stage1(img, imgPad, net_[0], classThreshold_[0]);
    winList = NMS(winList, true, nmsThreshold_[0]);

    winList = Stage2(imgPad, img180, net_[1], classThreshold_[1], 24, winList);
    winList = NMS(winList, true, nmsThreshold_[1]);

    winList = Stage3(imgPad, img180, img90, imgNeg90, net_[2], classThreshold_[2], 48, winList);
    winList = NMS(winList, false, nmsThreshold_[2]);
    winList = DeleteFP(winList);
    return winList;
}

std::vector<Window2> Impl::Track(cv::Mat img, cv::dnn::Net &net,
                                 float thres, int dim, std::vector<Window2> &winList)
{
    std::vector<cv::String> outputBlobNames = { "bbox_reg", "cls_prob", "points_reg", "rotate_reg" };

    if (winList.size() == 0)
        return winList;
    std::vector<Window> tmpWinList;
    for (size_t i = 0; i < winList.size(); i++)
    {
        Window win(winList[i].x - augScale_ * winList[i].w,
                   winList[i].y - augScale_ * winList[i].w,
                   winList[i].w + 2 * augScale_ * winList[i].w, winList[i].angle, winList[i].conf, winList[i].points14);
        tmpWinList.push_back(win);
    }
    std::vector<cv::Mat> dataList;
    for (size_t i = 0; i < tmpWinList.size(); i++)
    {
        dataList.push_back(PreProcessImg(CropFace(img, tmpWinList[i], dim), dim));
    }

    std::vector<cv::Mat> outputBlobs;

    std::vector<Window2> ret;

    for (size_t i = 0; i < tmpWinList.size(); i++)
    {
        cv::Mat inputBlob = cv::dnn::blobFromImage(dataList[i], 1.0, cv::Size(), cv::Scalar(), false, false);

        net.setInput(inputBlob);
        net.forward(outputBlobs, outputBlobNames);

        cv::Mat regression       = cv::Mat(outputBlobs[0].size[1], outputBlobs[0].size[0], CV_32F, outputBlobs[0].ptr<float>(0,0));
        cv::Mat prob             = cv::Mat(outputBlobs[1].size[1], outputBlobs[1].size[0], CV_32F, outputBlobs[1].ptr<float>(0,0));
        cv::Mat pointsRegression = cv::Mat(outputBlobs[2].size[1], outputBlobs[2].size[0], CV_32F, outputBlobs[2].ptr<float>(0,0));
        cv::Mat rotateProbs      = cv::Mat(outputBlobs[3].size[1], outputBlobs[3].size[0], CV_32F, outputBlobs[3].ptr<float>(0,0));

        float score = prob.at<float>(1, 0);

        if (score > thres)
        {
            float cropX = tmpWinList[i].x;
            float cropY = tmpWinList[i].y;
            float cropW = tmpWinList[i].width;
            float centerX = (2 * tmpWinList[i].x + tmpWinList[i].width - 1) / 2;
            float centerY = (2 * tmpWinList[i].y + tmpWinList[i].width - 1) / 2;
            std::vector<cv::Point> points14;

            for (int j = 0; j < pointsRegression.rows / 2; j++)
            {

                points14.push_back(RotatePoint((pointsRegression.at<float>(2 * j, 0) + 0.5) * (cropW - 1) + cropX,
                                               (pointsRegression.at<float>(2 * j + 1, 0) + 0.5) * (cropW - 1) + cropY,
                                               centerX, centerY, tmpWinList[i].angle));
            }

            float sn = regression.at<float>(0, 0);
            float xn = regression.at<float>(0, 1);
            float yn = regression.at<float>(0, 2);
            float theta = -tmpWinList[i].angle * M_PI / 180;
            int w = sn * cropW;
            int x = cropX  - 0.5 * sn * cropW +
                    cropW * sn * xn * std::cos(theta) - cropW * sn * yn * std::sin(theta) + 0.5 * cropW;
            int y = cropY  - 0.5 * sn * cropW +
                    cropW * sn * xn * std::sin(theta) + cropW * sn * yn * std::cos(theta) + 0.5 * cropW;

            float angle = angleRange_ * rotateProbs.at<float>(0, 0);
            if (thres > 0)
            {
                if (Legal(x, y, img) && Legal(x + w - 1, y + w - 1, img))
                {
                    int tmpW = w / (1 + 2 * augScale_);
                    if (tmpW >= 20)
                    {
                        ret.push_back(Window2(x + augScale_ * tmpW,
                                              y + augScale_ * tmpW,
                                              tmpW, tmpW, winList[i].angle + angle, winList[i].scale, score));
                        ret[ret.size() - 1].points14 = points14;
                    }
                }
            }
            else
            {
                int tmpW = w / (1 + 2 * augScale_);
                ret.push_back(Window2(x + augScale_ * tmpW,
                                      y + augScale_ * tmpW,
                                      tmpW, tmpW, winList[i].angle + angle, winList[i].scale, score));
                ret[ret.size() - 1].points14 = points14;
            }
        }
    }
    return ret;
}

cv::Point RotatePoint(float x, float y, float centerX, float centerY, float angle)
{
    x -= centerX;
    y -= centerY;
    float theta = -angle * M_PI / 180;
    float rx = centerX + x * std::cos(theta) - y * std::sin(theta);
    float ry = centerY + x * std::sin(theta) + y * std::cos(theta);
    return cv::Point(rx, ry);
}

void DrawLine(cv::Mat img, std::vector<cv::Point> pointList)
{
    int width = 2;
    cv::line(img, pointList[0], pointList[1], CYAN, width);
    cv::line(img, pointList[1], pointList[2], CYAN, width);
    cv::line(img, pointList[2], pointList[3], CYAN, width);
    cv::line(img, pointList[3], pointList[0], BLUE, width);
}

void DrawFace(cv::Mat img, Window face)
{
    float x1 = face.x;
    float y1 = face.y;
    float x2 = face.width + face.x - 1;
    float y2 = face.width + face.y - 1;
    float centerX = (x1 + x2) / 2;
    float centerY = (y1 + y2) / 2;
    std::vector<cv::Point> pointList;
    pointList.push_back(RotatePoint(x1, y1, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x1, y2, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x2, y2, centerX, centerY, face.angle));
    pointList.push_back(RotatePoint(x2, y1, centerX, centerY, face.angle));
    DrawLine(img, pointList);
}

void DrawPoints(cv::Mat img, Window face)
{
    int width = 2;
    if (face.points14.size() == 14)
    {
        for (int i = 1; i <= 8; i++)
        {
            cv::line(img, face.points14[i - 1], face.points14[i], BLUE, width);
        }
        for (size_t i = 0; i < face.points14.size(); i++)
        {
            if (i <= 8)
                cv::circle(img, face.points14[i], width, CYAN, -1);
            else if (i <= 9)
                cv::circle(img, face.points14[i], width, GREEN, -1);
            else if (i <= 11)
                cv::circle(img, face.points14[i], width, PURPLE, -1);
            else
                cv::circle(img, face.points14[i], width, RED, -1);
        }
    }
}

cv::Mat CropFace(cv::Mat img, Window face, int cropSize)
{
    float x1 = face.x;
    float y1 = face.y;
    float x2 = face.width + face.x - 1;
    float y2 = face.width + face.y - 1;
    float centerX = (x1 + x2) / 2;
    float centerY = (y1 + y2) / 2;
    cv::Point2f srcTriangle[3];
    cv::Point2f dstTriangle[3];
    srcTriangle[0] = RotatePoint(x1, y1, centerX, centerY, face.angle);
    srcTriangle[1] = RotatePoint(x1, y2, centerX, centerY, face.angle);
    srcTriangle[2] = RotatePoint(x2, y2, centerX, centerY, face.angle);
    dstTriangle[0] = cv::Point(0, 0);
    dstTriangle[1] = cv::Point(0, cropSize - 1);
    dstTriangle[2] = cv::Point(cropSize - 1, cropSize - 1);
    cv::Mat rotMat = cv::getAffineTransform(srcTriangle, dstTriangle);
    cv::Mat ret;
    cv::warpAffine(img, ret, rotMat, cv::Size(cropSize, cropSize));
    return ret;
}


