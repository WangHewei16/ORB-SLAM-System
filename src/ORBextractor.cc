#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <numeric> 
#include "ORBextractor.h"


using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;


static float IC_Angle(const Mat& image, Point2f pt, const vector<int> & u_max)
{
    int sumRowMoments = 0, sumColMoments = 0;

    // Corrected variable name from 'point' to 'pt'
    const uchar* imageCenter = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        sumColMoments += u * imageCenter[u];

    int step = static_cast<int>(image.step1());
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        int rowSum = 0;
        // Corrected variable name from 'umax' to 'u_max'
        int maxOffset = u_max[v];
        for (int u = -maxOffset; u <= maxOffset; ++u)
        {
            int pixelValueAbove = imageCenter[u + v * step];
            int pixelValueBelow = imageCenter[u - v * step];
            rowSum += (pixelValueAbove - pixelValueBelow);
            sumColMoments += u * (pixelValueAbove + pixelValueBelow);
        }
        sumRowMoments += v * rowSum;
    }

    return fastAtan2(static_cast<float>(sumRowMoments), static_cast<float>(sumColMoments));
}


const float factorPI = static_cast<float>(CV_PI / 180.f);

inline int getRotatedValue(const uchar* center, const Point& point, float a, float b, int step) {
    return center[cvRound(point.x * b + point.y * a) * step + cvRound(point.x * a - point.y * b)];
}

static void computeOrbDescriptor(const KeyPoint& kpt, const Mat& img, const Point* pattern, uchar* desc) {
    float angle = static_cast<float>(kpt.angle) * factorPI;
    float a = cos(angle), b = sin(angle);

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    int step = static_cast<int>(img.step);

    for (int i = 0; i < 32; ++i, pattern += 16) {
        int t0, t1, val = 0;
        for (int j = 0; j < 8; ++j) {
            t0 = getRotatedValue(center, pattern[2 * j], a, b, step);
            t1 = getRotatedValue(center, pattern[2 * j + 1], a, b, step);
            val |= (t0 < t1) << j;
        }
        desc[i] = static_cast<uchar>(val);
    }
}

static int bit_pattern_31_[256 * 4] = {
    8, -3, 9, 5, 4, 2, 7, -12, -11, 9, -8, 2, 7, -12, 12, -13, 2, -13, 2, 12,
    1, -7, 1, 6, -2, -10, -2, -4, -13, -13, -11, -8, -13, -3, -12, -9, 10, 4, 11, 9,
    -13, -8, -8, -9, -11, 7, -9, 12, 7, 7, 12, 6, -4, -5, -3, 0, -13, 2, -12, -3,
    -9, 0, -7, 5, 12, -6, 12, -1, -3, 6, -2, 12, -6, -13, -4, -8, 11, -13, 12, -8,
    4, 7, 5, 1, 5, -3, 10, -3, 3, -7, 6, 12, -8, -7, -6, -2, -2, 11, -1, -10,
    -13, 12, -8, 10, -7, 3, -5, -3, -4, 2, -3, 7, -10, -12, -6, 11, 5, -12, 6, -7,
    5, -6, 7, -1, 1, 0, 4, -5, 9, 11, 11, -13, 4, 7, 4, 12, 2, -1, 4, 4,
    -4, -12, -2, 7, -8, -5, -7, -10, 4, 11, 9, 12, 0, -8, 1, -13, -13, -2, -8, 2,
    -3, -2, -2, 3, -6, 9, -4, -9, 8, 12, 10, 7, 0, 9, 1, 3, 7, -5, 11, -10,
    -13, -6, -11, 0, 10, 7, 12, 1, -6, -3, -6, 12, 10, -9, 12, -4, -13, 8, -8, -12,
    -13, 0, -8, -4, 3, 3, 7, 8, 5, 7, 10, -7, -1, 7, 1, -12, 3, -10, 5, 6,
    2, -4, 3, -10, -13, 0, -13, 5, -13, -7, -12, 12, -13, 3, -11, 8, -7, 12, -4, 7,
    6, -10, 12, 8, -9, -1, -7, -6, -2, -5, 0, 12, -12, 5, -7, 5, 3, -10, 8, -13,
    -7, -7, -4, 5, -3, -2, -1, -7, 2, 9, 5, -11, -11, -13, -5, -13, -1, 6, 0, -1,
    5, -3, 5, 2, -4, -13, -4, 12, -9, -6, -9, 6, -12, -10, -8, -4, 10, 2, 12, -3,
    7, 12, 12, 12, -7, -13, -6, 5, -4, 9, -3, 4, 7, -1, 12, 2, -7, 6, -5, 1,
    -13, 11, -12, 5, -3, 7, -2, -6, 7, -8, 12, -7, -13, -7, -11, -12, 1, -3, 12, 12,
    2, -6, 3, 0, -4, 3, -2, -13, -1, -13, 1, 9, 7, 1, 8, -6, 1, -1, 3, 12,
    9, 1, 12, 6, -1, -9, -1, 3, -13, -13, -10, 5, 7, 7, 10, 12, 12, -5, 12, 9,
    6, 3, 7, 11, 5, -13, 6, 10, 2, -12, 2, 3, 3, 8, 4, -6, 2, 6, 12, -13,
    9, -12, 10, 3, -8, 4, -7, 9, -13, 6, 0, 11, -13, -1, -13, 1, 5, 5, 10, 8,
    0, -4, 2, 8, -9, 12, -5, -13, 0, 7, 2, 12, -1, 2, 1, 7, 5, 11, 7, -9,
    3, 5, 6, -8, -13, -4, -8, 9, -5, 9, -3, -3, -4, -7, -3, -12, 6, 5, 8, 0,
    -7, 6, -6, 12, -13, 6, -5, -2, 1, -10, 3, 10, 4, 1, 8, -4, -2, -2, 2, -13,
    2, -12, 12, 12, -2, -13, 0, -6, 4, 1, 9, 3, -6, -10, -3, -5, -3, -13, -1, 1,
    7, 5, 12, -11, 4, -2, 5, -7, -13, 9, -9, -5, 7, 1, 8, 6, 7, -8, 7, 6,
    -7, -4, -7, 1, -8, 11, -7, -8, -13, 6, -12, -8, 2, 4, 3, 9, 10, -5, 12, 3,
    -6, -5, -6, 7, 8, -3, 9, -8, 2, -12, 2, 8, -11, -2, -10, 3, -12, -13, -7, -9,
    -11, 0, -10, -5, 5, -3, 11, 8, -2, -13, -1, 12, -1, -8, 0, 9, -13, -11, -12, -5,
    -10, -2, -10, 11, -3, 9, -2, -13, 2, -3, 3, 2, -9, -13, -4, 0, -13, 5, 6, 0,
    11, -13, 12, -1, 9, 12, 10, -1, 5, -8, 10, -9, -1, 11, 1, -13, -9, -3, -6, 2,
    -1, -10, 1, 12, -13, 1, -8, -10, -8, -11, -10, -5, -13, 7, -11, 1, -13, 12, -11, -13,
    6, 0, 11, -13, 0, -1, 1, 4, -13, 3, -9, -2, -9, 8, -6, -3, -13, -6, -8, -2,
    5, -9, 8, 10, 2, 7, 3, -9, -1, -6, -1, -1, 9, 5, 11, -2, 11, -3, 12, -8,
    3, 0, 3, 5, -1, 4, 0, 10, 3, -6, 4, 5, -13, 0, -10, 5, 5, 8, 12, 11,
    8, 9, 9, -6, 7, -4, 8, -12, -10, 4, -10, 9, 7, 3, 12, 4, 9, -7, 10, -2,
    7, 0, 12, -2, -1, -6, 0, -11
};


ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST)
    : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    iniThFAST(_iniThFAST), minThFAST(_minThFAST) {

    mvScaleFactor.resize(nlevels, 1.0f);
    mvLevelSigma2.resize(nlevels, 1.0f);

    std::partial_sum(mvScaleFactor.begin(), mvScaleFactor.end() - 1, mvScaleFactor.begin() + 1,
        [this](float a, float b) { return a * this->scaleFactor; });

    std::transform(mvScaleFactor.begin(), mvScaleFactor.end(), mvLevelSigma2.begin(),
        [](float val) { return val * val; });

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);

    std::transform(mvScaleFactor.begin(), mvScaleFactor.end(), mvInvScaleFactor.begin(),
        [](float val) { return 1.0f / val; });

    std::transform(mvLevelSigma2.begin(), mvLevelSigma2.end(), mvInvLevelSigma2.begin(),
        [](float val) { return 1.0f / val; });

    mvImagePyramid.resize(nlevels);
    mnFeaturesPerLevel.resize(nlevels);

    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - pow(factor, nlevels));

    int sumFeatures = 0;
    std::generate_n(mnFeaturesPerLevel.begin(), nlevels - 1, [&]() {
        int currentFeatures = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += currentFeatures;
        nDesiredFeaturesPerScale *= factor;
        return currentFeatures;
        });
    mnFeaturesPerLevel.back() = std::max(nfeatures - sumFeatures, 0);

    std::copy_n((const Point*)bit_pattern_31_, 512, std::back_inserter(pattern));

    umax.resize(HALF_PATCH_SIZE + 1);

    int vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;

    for (int v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    for (int v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v) {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax) {
    for (auto& keypoint : keypoints) {
        keypoint.angle = IC_Angle(image, keypoint.pt, umax);
    }
}

void ExtractorNode::DivideNode(ExtractorNode& n1, ExtractorNode& n2, ExtractorNode& n3, ExtractorNode& n4) {
    int halfX = (UR.x - UL.x) / 2;
    int halfY = (BR.y - UL.y) / 2;

    // Assuming the constructor or assignment operator handles these initializations:
    n1.UL = UL;
    n1.UR = {UL.x + halfX, UL.y};
    n1.BL = {UL.x, UL.y + halfY};
    n1.BR = {UL.x + halfX, UL.y + halfY};

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = {UR.x, UL.y + halfY};

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = {n1.BR.x, BL.y};

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;

    // Reserve space for keypoints based on parent node
    n1.vKeys.reserve(vKeys.size());
    n2.vKeys.reserve(vKeys.size());
    n3.vKeys.reserve(vKeys.size());
    n4.vKeys.reserve(vKeys.size());

    // Distribute keypoints
    for (const auto& kp : vKeys) {
        if (kp.pt.x < n1.UR.x) {
            if (kp.pt.y < n1.BR.y) n1.vKeys.push_back(kp);
            else n3.vKeys.push_back(kp);
        } else {
            if (kp.pt.y < n1.BR.y) n2.vKeys.push_back(kp);
            else n4.vKeys.push_back(kp);
        }
    }

    // Mark nodes with only one keypoint as not to be divided further
    n1.bNoMore = n1.vKeys.size() == 1;
    n2.bNoMore = n2.vKeys.size() == 1;
    n3.bNoMore = n3.vKeys.size() == 1;
    n4.bNoMore = n4.vKeys.size() == 1;
}


vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int& minX,
    const int& maxX, const int& minY, const int& maxY, const int& N, const int& level) {
    int nIni = (maxX - minX) / (maxY - minY);
    float hX = static_cast<float>(maxX - minX) / nIni;

    list<ExtractorNode> lNodes;
    vector<ExtractorNode*> vpIniNodes(nIni);

    for (int i = 0; i < nIni; ++i) {
        ExtractorNode ni;
        ni.UL = cv::Point2i(int(hX * i), 0);
        ni.UR = cv::Point2i(int(hX * (i + 1)), 0);
        ni.BL = cv::Point2i(int(hX * i), maxY - minY);
        ni.BR = cv::Point2i(int(hX * (i + 1)), maxY - minY);
        ni.vKeys.reserve(vToDistributeKeys.size());
        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    for (const auto& kp : vToDistributeKeys) {
        int idx = static_cast<int>(kp.pt.x / hX);
        if (idx >= 0 && idx < nIni) {
            vpIniNodes[idx]->vKeys.push_back(kp);
        }
    }

    bool finish = false;
    while (!finish) {
        for (auto lit = lNodes.begin(); lit != lNodes.end();) {
            if (lit->vKeys.size() == 1) {
                lit->bNoMore = true;
                ++lit;
            } else if (lit->vKeys.empty()) {
                lit = lNodes.erase(lit);
            } else {
                ExtractorNode n1, n2, n3, n4;
                lit->DivideNode(n1, n2, n3, n4);
                if (!n1.vKeys.empty()) lNodes.push_front(n1);
                if (!n2.vKeys.empty()) lNodes.push_front(n2);
                if (!n3.vKeys.empty()) lNodes.push_front(n3);
                if (!n4.vKeys.empty()) lNodes.push_front(n4);
                lit = lNodes.erase(lit);
            }
        }
        finish = (lNodes.size() >= N || all_of(lNodes.begin(), lNodes.end(), [](const ExtractorNode& n) { return n.bNoMore; }));
    }

    vector<cv::KeyPoint> resultKeys;
    resultKeys.reserve(N);
    for (const auto& node : lNodes) {
        if (!node.vKeys.empty()) {
            auto maxIt = max_element(node.vKeys.begin(), node.vKeys.end(), [](const KeyPoint& a, const KeyPoint& b) {
                return a.response < b.response;
            });
            resultKeys.push_back(*maxIt);
        }
    }
    return resultKeys;
}

void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint>>& allKeypoints)
{
    const float W = 30;
    allKeypoints.resize(nlevels);

    auto handleKeyPoints = [this](const Mat& image, vector<KeyPoint>& keys, int threshold) {
        if (keys.empty())
            FAST(image, keys, threshold, true);
        };

    for (int level = 0; level < nlevels; ++level)
    {
        int minBorderX = EDGE_THRESHOLD - 3;
        int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
        int minBorderY = minBorderX;
        int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

        vector<KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(nfeatures * 10);

        float width = maxBorderX - minBorderX;
        float height = maxBorderY - minBorderY;

        int nCols = width / W;
        int nRows = height / W;
        int wCell = ceil(width / nCols);
        int hCell = ceil(height / nRows);

        for (int i = 0; i < nRows; ++i)
        {
            float iniY = minBorderY + i * hCell;
            float maxY = std::min(iniY + hCell + 6, static_cast<float>(maxBorderY));

            for (int j = 0; j < nCols; ++j)
            {
                float iniX = minBorderX + j * wCell;
                float maxX = std::min(iniX + wCell + 6, static_cast<float>(maxBorderX));

                Rect rect(iniX, iniY, maxX - iniX, maxY - iniY);
                vector<KeyPoint> vKeysCell;
                Mat cellImage = mvImagePyramid[level](rect);

                FAST(cellImage, vKeysCell, iniThFAST, true);
                handleKeyPoints(cellImage, vKeysCell, minThFAST);

                for (auto& key : vKeysCell)
                {
                    key.pt.x += j * wCell;
                    key.pt.y += i * hCell;
                    vToDistributeKeys.push_back(key);
                }
            }
        }

        vector<KeyPoint>& keypoints = allKeypoints[level];
        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX, minBorderY, maxBorderY, mnFeaturesPerLevel[level], level);

        int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];
        for (auto& keypoint : keypoints)
        {
            keypoint.pt.x += minBorderX;
            keypoint.pt.y += minBorderY;
            keypoint.octave = level;
            keypoint.size = scaledPatchSize;
        }
    }

    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

void ORBextractor::ComputeKeyPointsOld(vector<vector<KeyPoint>>& allKeypoints)
{
    allKeypoints.resize(nlevels);
    float imageRatio = static_cast<float>(mvImagePyramid[0].cols) / mvImagePyramid[0].rows;

    for (int level = 0; level < nlevels; ++level)
    {
        const int nDesiredFeatures = mnFeaturesPerLevel[level];
        const int levelCols = sqrt(static_cast<float>(nDesiredFeatures) / (5 * imageRatio));
        const int levelRows = static_cast<int>(imageRatio * levelCols);

        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD;
        const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD;

        const int cellWidth = ceil(static_cast<float>(maxBorderX - minBorderX) / levelCols);
        const int cellHeight = ceil(static_cast<float>(maxBorderY - minBorderY) / levelRows);

        vector<vector<vector<KeyPoint>>> cellKeyPoints(levelRows, vector<vector<KeyPoint>>(levelCols));

        for (int i = 0; i < levelRows; ++i)
        {
            for (int j = 0; j < levelCols; ++j)
            {
                int iniX = minBorderX + j * cellWidth - 3;
                int iniY = minBorderY + i * cellHeight - 3;
                int maxX = iniX + cellWidth + 6;
                int maxY = iniY + cellHeight + 6;

                Rect cellRect(iniX, iniY, min(maxX, maxBorderX) - iniX, min(maxY, maxBorderY) - iniY);
                cellKeyPoints[i][j].reserve(nDesiredFeatures);

                FAST(mvImagePyramid[level](cellRect), cellKeyPoints[i][j], iniThFAST, true);
                if (cellKeyPoints[i][j].size() <= 3)
                {
                    cellKeyPoints[i][j].clear();
                    FAST(mvImagePyramid[level](cellRect), cellKeyPoints[i][j], minThFAST, true);
                }

                for (auto& key : cellKeyPoints[i][j])
                {
                    key.pt.x += iniX;
                    key.pt.y += iniY;
                    key.octave = level;
                    key.size = PATCH_SIZE * mvScaleFactor[level];
                }
            }
        }

        vector<KeyPoint>& keypoints = allKeypoints[level];
        keypoints.clear();
        keypoints.reserve(nDesiredFeatures);

        for (const auto& row : cellKeyPoints)
        {
            for (const auto& col : row)
            {
                keypoints.insert(keypoints.end(), col.begin(), col.end());
            }
        }

        if (static_cast<int>(keypoints.size()) > nDesiredFeatures)
        {
            KeyPointsFilter::retainBest(keypoints, nDesiredFeatures);
            keypoints.resize(nDesiredFeatures);
        }
    }

    for (int level = 0; level < nlevels; ++level)
    {
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
    }
}

static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors, const vector<Point>& pattern)
{
    descriptors.create(static_cast<int>(keypoints.size()), 32, CV_8UC1);
    for (int i = 0; i < keypoints.size(); ++i) {
        computeOrbDescriptor(keypoints[i], image, pattern.data(), descriptors.ptr<uint8_t>(i));
    }
}

void ORBextractor::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints, OutputArray _descriptors)
{
    if (_image.empty())
        return;

    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1);

    ComputePyramid(image);

    vector<vector<KeyPoint>> allKeypoints;
    ComputeKeyPointsOctTree(allKeypoints);

    int nkeypoints = 0;
    for (const auto& kpts : allKeypoints) {
        nkeypoints += kpts.size();
    }

    if (nkeypoints == 0) {
        _descriptors.release();
        return;
    }

    _descriptors.create(nkeypoints, 32, CV_8U);
    Mat descriptors = _descriptors.getMat();

    _keypoints.clear();
    _keypoints.reserve(nkeypoints);

    int offset = 0;
    for (int level = 0; level < nlevels; ++level)
    {
        auto& keypoints = allKeypoints[level];
        if (keypoints.empty())
            continue;

        Mat workingMat = mvImagePyramid[level].clone();
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        Mat desc = descriptors.rowRange(offset, offset + keypoints.size());
        computeDescriptors(workingMat, keypoints, desc, pattern);

        offset += keypoints.size();

        if (level != 0) {
            float scale = mvScaleFactor[level];
            for (auto& keypoint : keypoints) {
                keypoint.pt *= scale;
            }
        }

        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}

void ORBextractor::ComputePyramid(Mat image)
{
    for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];
        Size sz(cvRound(image.cols * scale), cvRound(image.rows * scale));
        Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
        Mat temp(wholeSize, image.type());
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        if (level == 0) {
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, BORDER_REFLECT_101);
        }
        else {
            resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);
            copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, BORDER_REFLECT_101 + BORDER_ISOLATED);
        }
    }
}
}