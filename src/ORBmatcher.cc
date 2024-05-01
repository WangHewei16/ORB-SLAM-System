#include "ORBmatcher.h"
#include <limits.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include <stdint-gcc.h>

using namespace std;
using namespace cv;
namespace ORB_SLAM2
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri) {}

int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th) {
    int nmatches = 0;
    bool bFactor = th != 1.0;

    for (auto& pMP : vpMapPoints) {
        if (!pMP || pMP->isBad() || !pMP->mbTrackInView) continue;

        int nPredictedLevel = pMP->mnTrackScaleLevel;
        float r = RadiusByViewingCos(pMP->mTrackViewCos) * (bFactor ? th : 1);
        auto vIndices = F.GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY, r * F.mvScaleFactors[nPredictedLevel], nPredictedLevel - 1, nPredictedLevel);

        if (vIndices.empty()) continue;

        const cv::Mat MPdescriptor = pMP->GetDescriptor();
        int bestDist = INT_MAX;
        int bestIdx = -1;
        int secondBestDist = INT_MAX;

        for (auto idx : vIndices) {
            if (F.mvpMapPoints[idx] && F.mvpMapPoints[idx]->Observations() > 0)
                continue;

            if (F.mvuRight[idx] > 0) {
                float er = fabs(pMP->mTrackProjXR - F.mvuRight[idx]);
                if (er > r * F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            int dist = DescriptorDistance(MPdescriptor, F.mDescriptors.row(idx));

            if (dist < bestDist) {
                secondBestDist = bestDist;
                bestDist = dist;
                bestIdx = idx;
            } else if (dist < secondBestDist) {
                secondBestDist = dist;
            }
        }

        if (bestDist <= TH_HIGH && (bestDist <= mfNNratio * secondBestDist)) {
            F.mvpMapPoints[bestIdx] = pMP;
            nmatches++;
        }
    }

    return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos) {
    return (viewCos > 0.998) ? 2.5f : 4.0f;
}

bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame* pKF2) {
    const float a = kp1.pt.x * F12.at<float>(0, 0) + kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
    const float b = kp1.pt.x * F12.at<float>(0, 1) + kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
    const float c = kp1.pt.x * F12.at<float>(0, 2) + kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

    const float num = a * kp2.pt.x + b * kp2.pt.y + c;
    const float den = a * a + b * b;

    if (den == 0) {
        return false;
    }

    const float dsqr = num * num / den;
    return dsqr < 3.84 * pKF2->mvLevelSigma2[kp2.octave];
}


int ORBmatcher::SearchByBoW(KeyFrame* pKF, Frame &F, vector<MapPoint*>& vpMapPointMatches) {
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();
    vpMapPointMatches.resize(F.N, nullptr);

    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;
    int nmatches = 0;

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++) {
        rotHist[i].reserve(500);
    }
    const float factor = 1.0f / HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while (KFit != KFend && Fit != Fend) {
        if (KFit->first == Fit->first) {
            // processing nodes with matching vocabulary ids
            KFit++;
            Fit++;
        } else if (KFit->first < Fit->first) {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        } else {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*>& vpPoints, vector<MapPoint*>& vpMatched, int th) {
    const float &fx = pKF->fx, &fy = pKF->fy;
    const float &cx = pKF->cx, &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
    float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw / scw;
    cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
    cv::Mat Ow = -Rcw.t() * tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(nullptr);

    int nmatches = 0;

    for (auto& pMP : vpPoints) {
        if (pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc = Rcw * p3Dw + tcw;

        if (p3Dc.at<float>(2) < 0.0f) continue;  // Depth must be positive

        float invz = 1.0f / p3Dc.at<float>(2);
        float u = fx * p3Dc.at<float>(0) * invz + cx;
        float v = fy * p3Dc.at<float>(1) * invz + cy;
        if (!pKF->IsInImage(u, v)) continue;

        // Depth must be within the scale pyramid of the MapPoint
        float maxDistance = pMP->GetMaxDistanceInvariance();
        float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw - Ow;
        float dist = cv::norm(PO);

        if (dist < minDistance || dist > maxDistance)
            continue;

        // Viewing angle must be less than 60 degrees
        cv::Mat Pn = pMP->GetNormal();
        if (PO.dot(Pn) < 0.5 * dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist, pKF);
        float radius = th * pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);
        if (vIndices.empty()) continue;

        const cv::Mat MPdescriptor = pMP->GetDescriptor();
        int bestDist = INT_MAX;
        int bestIdx = -1;

        for (auto idx : vIndices) {
            if (vpMatched[idx]) continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);
            int dist = DescriptorDistance(MPdescriptor, dKF);

            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_LOW) {
            vpMatched[bestIdx] = pMP;
            nmatches++;
        }
    }

    return nmatches;
}

int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize) {
    int nmatches = 0;
    vnMatches12.resize(F1.mvKeysUn.size(), -1);

    vector<int> rotHist[HISTO_LENGTH];
    for (auto &hist : rotHist) {
        hist.reserve(500);
    }
    const float factor = 1.0f / HISTO_LENGTH;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(), INT_MAX);
    vector<int> vnMatches21(F2.mvKeysUn.size(), -1);

    for (size_t i1 = 0; i1 < F1.mvKeysUn.size(); ++i1) {
        const cv::KeyPoint &kp1 = F1.mvKeysUn[i1];
        if (kp1.octave > 0) continue;

        auto vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize, kp1.octave, kp1.octave);
        if (vIndices2.empty()) continue;

        const cv::Mat &d1 = F1.mDescriptors.row(i1);
        int bestDist = INT_MAX, bestDist2 = INT_MAX, bestIdx2 = -1;

        for (auto i2 : vIndices2) {
            const cv::Mat &d2 = F2.mDescriptors.row(i2);
            int dist = DescriptorDistance(d1, d2);

            if (dist < vMatchedDistance[i2]) {
                if (dist < bestDist) {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestIdx2 = i2;
                } else if (dist < bestDist2) {
                    bestDist2 = dist;
                }
            }
        }

        if (bestDist <= TH_LOW && bestDist < static_cast<float>(bestDist2) * mfNNratio) {
            if (vnMatches21[bestIdx2] >= 0) {
                vnMatches12[vnMatches21[bestIdx2]] = -1;
                nmatches--;
            }
            vnMatches12[i1] = bestIdx2;
            vnMatches21[bestIdx2] = i1;
            vMatchedDistance[bestIdx2] = bestDist;
            nmatches++;

            if (mbCheckOrientation) {
                float rot = kp1.angle - F2.mvKeysUn[bestIdx2].angle;
                if (rot < 0.0) rot += 360.0f;
                int bin = round(rot * factor);
                if (bin == HISTO_LENGTH) bin = 0;
                rotHist[bin].push_back(i1);
            }
        }
    }

    if (mbCheckOrientation) {
        int ind1 = -1, ind2 = -1, ind3 = -1;
        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i) {
            if (i == ind1 || i == ind2 || i == ind3) continue;
            for (int idx1 : rotHist[i]) {
                if (vnMatches12[idx1] >= 0) {
                    vnMatches12[idx1] = -1;
                    nmatches--;
                }
            }
        }
    }

    for (size_t i1 = 0; i1 < vnMatches12.size(); ++i1) {
        if (vnMatches12[i1] >= 0)
            vbPrevMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;
    }

    return nmatches;
}

int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12) {
    const auto &vKeysUn1 = pKF1->mvKeysUn;
    const auto &vFeatVec1 = pKF1->mFeatVec;
    const auto &vpMapPoints1 = pKF1->GetMapPointMatches();
    const auto &Descriptors1 = pKF1->mDescriptors;

    const auto &vKeysUn2 = pKF2->mvKeysUn;
    const auto &vFeatVec2 = pKF2->mFeatVec;
    const auto &vpMapPoints2 = pKF2->GetMapPointMatches();
    const auto &Descriptors2 = pKF2->mDescriptors;

    vpMatches12.resize(vpMapPoints1.size(), nullptr);
    vector<bool> vbMatched2(vpMapPoints2.size(), false);

    vector<int> rotHist[HISTO_LENGTH];
    for (auto &hist : rotHist) {
        hist.reserve(500);
    }
    const float factor = 1.0f / HISTO_LENGTH;

    int nmatches = 0;

    auto f1it = vFeatVec1.begin();
    auto f2it = vFeatVec2.begin();
    auto f1end = vFeatVec1.end();
    auto f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end) {
        if (f1it->first == f2it->first) {
            for (size_t idx1 : f1it->second) {
                MapPoint* pMP1 = vpMapPoints1[idx1];
                if (!pMP1 || pMP1->isBad()) continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);
                int bestDist1 = INT_MAX, bestIdx2 = -1, bestDist2 = INT_MAX;

                for (size_t idx2 : f2it->second) {
                    MapPoint* pMP2 = vpMapPoints2[idx2];
                    if (vbMatched2[idx2] || !pMP2 || pMP2->isBad()) continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);
                    int dist = DescriptorDistance(d1, d2);

                    if (dist < bestDist1) {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdx2 = idx2;
                    } else if (dist < bestDist2) {
                        bestDist2 = dist;
                    }
                }
                if (bestDist1 < TH_LOW && static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2)) {
                    vpMatches12[idx1] = vpMapPoints2[bestIdx2];
                    vbMatched2[bestIdx2] = true;
                    nmatches++;

                    if (mbCheckOrientation) {
                        float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
                        if (rot < 0.0f) rot += 360.0f;
                        int bin = round(rot * factor);
                        if (bin == HISTO_LENGTH) bin = 0;
                        rotHist[bin].push_back(idx1);
                    }
                }
            }
            ++f1it;
            ++f2it;
        } else if (f1it->first < f2it->first) {
            f1it = vFeatVec1.lower_bound(f2it->first);
        } else {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if (mbCheckOrientation) {
        int ind1 = -1, ind2 = -1, ind3 = -1;
        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i) {
            if (i == ind1 || i == ind2 || i == ind3) continue;
            for (int idx1 : rotHist[i]) {
                vpMatches12[idx1] = nullptr;
                nmatches--;
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo) {
    const auto &vFeatVec1 = pKF1->mFeatVec;
    const auto &vFeatVec2 = pKF2->mFeatVec;

    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w * Cw + t2w;
    float invz = 1.0f / C2.at<float>(2);
    float ex = pKF2->fx * C2.at<float>(0) * invz + pKF2->cx;
    float ey = pKF2->fy * C2.at<float>(1) * invz + pKF2->cy;

    int nmatches = 0;
    vector<bool> vbMatched2(pKF2->N, false);
    vector<int> vMatches12(pKF1->N, -1);

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++) {
        rotHist[i].reserve(500);
    }
    float factor = 1.0f / HISTO_LENGTH;

    auto f1it = vFeatVec1.begin();
    auto f2it = vFeatVec2.begin();
    auto f1end = vFeatVec1.end();
    auto f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end) {
        if (f1it->first == f2it->first) {
            for (size_t idx1 : f1it->second) {
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                if (pMP1 || (bOnlyStereo && pKF1->mvuRight[idx1] < 0)) continue;

                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                int bestDist = TH_LOW;
                int bestIdx2 = -1;

                for (size_t idx2 : f2it->second) {
                    if (vbMatched2[idx2] || pKF2->GetMapPoint(idx2) || (bOnlyStereo && pKF2->mvuRight[idx2] < 0)) continue;

                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    int dist = DescriptorDistance(d1, d2);

                    if (dist > TH_LOW || dist > bestDist) continue;

                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
                    if (!bOnlyStereo && CheckDistEpipolarLine(kp1, kp2, F12, pKF2)) {
                        bestIdx2 = idx2;
                        bestDist = dist;
                        if (mbCheckOrientation) {
                            float rot = kp1.angle - kp2.angle;
                            if(rot < 0.0)
                                rot += 360.0f;
                            int bin = static_cast<int>(round(rot * factor)) % HISTO_LENGTH;
                            rotHist[bin].push_back(idx1);
                        }
                    }
                }

                if (bestIdx2 >= 0) {
                    vMatches12[idx1] = bestIdx2;
                    nmatches++;
                }
            }
            ++f1it;
            ++f2it;
        } else if (f1it->first < f2it->first) {
            f1it = vFeatVec1.lower_bound(f2it->first);
        } else {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if (mbCheckOrientation) {
        int ind1 = -1, ind2 = -1, ind3 = -1;
        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++) {
            if (i != ind1 && i != ind2 && i != ind3) {
                for (size_t idx : rotHist[i]) {
                    if (vMatches12[idx] >= 0) {
                        vMatches12[idx] = -1;
                        nmatches--;
                    }
                }
            }
        }
    }
    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);
    for (size_t i = 0; i < vMatches12.size(); i++) {
        if (vMatches12[i] >= 0) {
            vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
        }
    }
    return nmatches;
}

void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3) {
    int topIndices[3] = {-1, -1, -1};
    int topValues[3] = {0, 0, 0};

    auto insertInTop = [&](int idx, int value) {
        for (int j = 0; j < 3; ++j) {
            if (value > topValues[j]) {
                for (int k = 2; k > j; --k) {
                    topValues[k] = topValues[k-1];
                    topIndices[k] = topIndices[k-1];
                }
                topValues[j] = value;
                topIndices[j] = idx;
                break;
            }
        }
    };

    for (int i = 0; i < L; ++i) {
        const int size = histo[i].size();
        insertInTop(i, size);
    }

    ind1 = topIndices[0];
    ind2 = topIndices[1];
    ind3 = topIndices[2];

    if (topValues[1] < 0.1f * topValues[0]) {
        ind2 = -1;
        ind3 = -1;
    } else if (topValues[2] < 0.1f * topValues[0]) {
        ind3 = -1;
    }
}

int ORBmatcher::Fuse(KeyFrame* pKF, const vector<MapPoint*>& vpMapPoints, const float th) {
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();
    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused = 0;

    for (auto pMP : vpMapPoints) {
        if (!pMP || pMP->isBad() || pMP->IsInKeyFrame(pKF)) continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc = Rcw * p3Dw + tcw;

        if (p3Dc.at<float>(2) < 0.0f) continue; // Depth must be positive

        float invz = 1.0 / p3Dc.at<float>(2);
        float x = p3Dc.at<float>(0) * invz;
        float y = p3Dc.at<float>(1) * invz;
        float u = pKF->fx * x + pKF->cx;
        float v = pKF->fy * y + pKF->cy;

        if (!pKF->IsInImage(u, v)) continue;

        float dist = cv::norm(p3Dw - Ow);
        if (dist < pMP->GetMinDistanceInvariance() || dist > pMP->GetMaxDistanceInvariance()) continue;

        int level = pMP->PredictScale(dist, pKF);
        float radius = th * pKF->mvScaleFactors[level];
        auto indices = pKF->GetFeaturesInArea(u, v, radius);

        if (indices.empty()) continue;

        const cv::Mat& MPdescriptor = pMP->GetDescriptor();
        int bestDist = INT_MAX;
        int bestIdx = -1;

        for (auto idx : indices) {
            const cv::KeyPoint& kp = pKF->mvKeysUn[idx];
            if (kp.octave < level - 1 || kp.octave > level) continue;

            const cv::Mat& dKF = pKF->mDescriptors.row(idx);
            int dist = DescriptorDistance(MPdescriptor, dKF);

            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_LOW) {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if (pMPinKF) {
                if (!pMPinKF->isBad()) {
                    if (pMPinKF->Observations() > pMP->Observations())
                        pMP->Replace(pMPinKF);
                }
            } else {
                pMP->AddObservation(pKF, bestIdx);
                pKF->AddMapPoint(pMP, bestIdx);
            }
            nFused++;
        }
    }
    return nFused;
}

int ORBmatcher::Fuse(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*>& vpPoints, float th, vector<MapPoint*>& vpReplacePoint) {
    cv::Mat Rcw = Scw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Scw.rowRange(0, 3).col(3);
    cv::Mat Ow = -Rcw.t() * tcw;

    int nFused = 0;

    for (int iMP = 0; iMP < vpPoints.size(); iMP++) {
        MapPoint* pMP = vpPoints[iMP];

        if (!pMP || pMP->isBad() || pKF->GetMapPoint(pMP->GetIndexInKeyFrame(pKF))) continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc = Rcw * p3Dw + tcw;

        if (p3Dc.at<float>(2) < 0.0f) continue;

        float invz = 1.0 / p3Dc.at<float>(2);
        float x = p3Dc.at<float>(0) * invz;
        float y = p3Dc.at<float>(1) * invz;
        float u = pKF->fx * x + pKF->cx;
        float v = pKF->fy * y + pKF->cy;

        if (!pKF->IsInImage(u, v)) continue;

        float dist = cv::norm(p3Dw - Ow);
        if (dist < pMP->GetMinDistanceInvariance() || dist > pMP->GetMaxDistanceInvariance()) continue;

        int level = pMP->PredictScale(dist, pKF);
        float radius = th * pKF->mvScaleFactors[level];
        auto indices = pKF->GetFeaturesInArea(u, v, radius);

        if (indices.empty()) continue;

        const cv::Mat& dMP = pMP->GetDescriptor();
        int bestDist = INT_MAX;
        int bestIdx = -1;

        for (auto idx : indices) {
            const cv::KeyPoint& kp = pKF->mvKeysUn[idx];
            if (kp.octave < level - 1 || kp.octave > level) continue;

            const cv::Mat& dKF = pKF->mDescriptors.row(idx);
            int dist = DescriptorDistance(dMP, dKF);

            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_LOW) {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if (pMPinKF) {
                if (!pMPinKF->isBad()) vpReplacePoint[iMP] = pMPinKF;
            } else {
                pMP->AddObservation(pKF, bestIdx);
                pKF->AddMapPoint(pMP, bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int ORBmatcher::SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, vector<MapPoint*>& vpMatches12,
                             const float& s12, const cv::Mat& R12, const cv::Mat& t12, const float th) {
    // Calibration parameters
    const float& fx = pKF1->fx;
    const float& fy = pKF1->fy;
    const float& cx = pKF1->cx;
    const float& cy = pKF1->cy;

    // Extract rotation and translation from both keyframes
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    // Compute the transformation from the first to the second camera frame
    cv::Mat sR12 = s12 * R12;
    cv::Mat sR21 = (1.0 / s12) * R12.t();
    cv::Mat t21 = -sR21 * t12;

    // Map points from both keyframes
    const vector<MapPoint*>& vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<bool> vbAlreadyMatched1(vpMapPoints1.size(), false);
    vector<int> vnMatch1(vpMapPoints1.size(), -1);

    // Initialize the match vector for the first keyframe
    for (int i = 0; i < vpMatches12.size(); i++) {
        MapPoint* pMP = vpMatches12[i];
        if (pMP) {
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if (idx2 >= 0) vbAlreadyMatched1[i] = true;
        }
    }

    int nMatches = 0;

    for (int i1 = 0; i1 < vpMapPoints1.size(); ++i1) {
        MapPoint* pMP1 = vpMapPoints1[i1];

        if (!pMP1 || vbAlreadyMatched1[i1] || pMP1->isBad()) continue;

        cv::Mat p3Dw = pMP1->GetWorldPos();
        cv::Mat p3Dc1 = R1w * p3Dw + t1w;
        cv::Mat p3Dc2 = sR21 * p3Dc1 + t21;

        if (p3Dc2.at<float>(2) < 0.0f) continue; // Check if the point is in front of the camera

        float invz = 1.0f / p3Dc2.at<float>(2);
        float x = p3Dc2.at<float>(0) * invz;
        float y = p3Dc2.at<float>(1) * invz;
        float u = fx * x + cx;
        float v = fy * y + cy;

        if (!pKF2->IsInImage(u, v)) continue; // Check if the point projects inside the frame

        float dist = cv::norm(p3Dc2);
        if (dist < pMP1->GetMinDistanceInvariance() || dist > pMP1->GetMaxDistanceInvariance()) continue;

        int predictedLevel = pMP1->PredictScale(dist, pKF2);
        float radius = th * pKF2->mvScaleFactors[predictedLevel];

        const vector<size_t>& vIndices = pKF2->GetFeaturesInArea(u, v, radius);
        if (vIndices.empty()) continue;

        const cv::Mat& dMP1 = pMP1->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx2 = -1;
        for (auto idx : vIndices) {
            const cv::KeyPoint& kp = pKF2->mvKeysUn[idx];
            if (kp.octave < predictedLevel - 1 || kp.octave > predictedLevel) continue;

            const cv::Mat& dKF2 = pKF2->mDescriptors.row(idx);
            int dist = DescriptorDistance(dMP1, dKF2);

            if (dist < bestDist) {
                bestDist = dist;
                bestIdx2 = idx;
            }
        }

        if (bestDist <= TH_HIGH) {
            vnMatch1[i1] = bestIdx2;
            nMatches++;
        }
    }

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(), nullptr);
    for (int i = 0; i < vnMatch1.size(); i++) {
        if (vnMatch1[i] >= 0) {
            vpMatches12[i] = pKF2->GetMapPoint(vnMatch1[i]);
        }
    }

    return nMatches;
}

int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono) {
    int nmatches = 0;
    vector<int> rotHist[HISTO_LENGTH];
    for(int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f / HISTO_LENGTH;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat twc = -Rcw.t() * tcw;

    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat tlc = Rlw * twc + tlw;

    const bool bForward = tlc.at<float>(2) > CurrentFrame.mb && !bMono;
    const bool bBackward = -tlc.at<float>(2) > CurrentFrame.mb && !bMono;

    for(int i = 0; i < LastFrame.N; i++) {
        MapPoint* pMP = LastFrame.mvpMapPoints[i];
        if(!pMP) continue;

        if(!LastFrame.mvbOutlier[i]) {
            cv::Mat x3Dw = pMP->GetWorldPos();
            cv::Mat x3Dc = Rcw * x3Dw + tcw;
            const float z = x3Dc.at<float>(2);
            if(z <= 0) continue;

            float u = CurrentFrame.fx * x3Dc.at<float>(0) / z + CurrentFrame.cx;
            float v = CurrentFrame.fy * x3Dc.at<float>(1) / z + CurrentFrame.cy;

            if(u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX || v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
                continue;

            int nLastOctave = LastFrame.mvKeys[i].octave;
            float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];
            vector<size_t> vIndices2;
            if(bForward)
                vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave);
            else if(bBackward)
                vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, 0, nLastOctave);
            else
                vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave-1, nLastOctave+1);

            if(vIndices2.empty()) continue;

            const cv::Mat dMP = pMP->GetDescriptor();
            int bestDist = INT_MAX;
            int bestIdx2 = -1;

            for(auto idx : vIndices2) {
                if(CurrentFrame.mvpMapPoints[idx]) continue;
                const cv::Mat &d = CurrentFrame.mDescriptors.row(idx);
                int dist = DescriptorDistance(dMP, d);
                if(dist < bestDist) {
                    bestDist = dist;
                    bestIdx2 = idx;
                }
            }

            if(bestDist <= TH_HIGH) {
                CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
                nmatches++;
                if(mbCheckOrientation) {
                    float rot = LastFrame.mvKeys[i].angle - CurrentFrame.mvKeys[bestIdx2].angle;
                    int bin = static_cast<int>(round(rot * factor)) % HISTO_LENGTH;
                    rotHist[bin].push_back(bestIdx2);
                }
            }
        }
    }
    // Apply rotation consistency
    if(mbCheckOrientation) {
        int ind1 = -1, ind2 = -1, ind3 = -1;
        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for(int i = 0; i < HISTO_LENGTH; i++) {
            if(i != ind1 && i != ind2 && i != ind3) {
                for(auto idx : rotHist[i]) {
                    CurrentFrame.mvpMapPoints[idx] = nullptr;
                    nmatches--;
                }
            }
        }
    }
    return nmatches;
}

int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist) {
    int matchesCount = 0;
    vector<int> rotationHistogram[HISTO_LENGTH];
    std::fill(std::begin(rotationHistogram), std::end(rotationHistogram), vector<int>());
    float histogramFactor = 1.0f / HISTO_LENGTH;

    cv::Mat rotationCurrent = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat translationCurrent = CurrentFrame.mTcw.rowRange(0, 3).col(3);
    cv::Mat positionCurrent = -rotationCurrent.t() * translationCurrent;

    vector<MapPoint*> mapPoints = pKF->GetMapPointMatches();

    for (size_t i = 0; i < mapPoints.size(); ++i) {
        MapPoint* pMP = mapPoints[i];
        if (!pMP || pMP->isBad() || sAlreadyFound.count(pMP)) continue;

        cv::Mat worldPos = pMP->GetWorldPos();
        cv::Mat cameraPos = rotationCurrent * worldPos + translationCurrent;
        float invZ = 1.0 / cameraPos.at<float>(2);
        float u = CurrentFrame.fx * cameraPos.at<float>(0) * invZ + CurrentFrame.cx;
        float v = CurrentFrame.fy * cameraPos.at<float>(1) * invZ + CurrentFrame.cy;

        if(u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX || v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
            continue;

        float dist3D = cv::norm(worldPos - positionCurrent); 
        int predictedLevel = pMP->PredictScale(dist3D, &CurrentFrame);
        float radius = th * CurrentFrame.mvScaleFactors[predictedLevel];
        vector<size_t> candidates = CurrentFrame.GetFeaturesInArea(u, v, radius, predictedLevel - 1, predictedLevel + 1);

        if (candidates.empty()) continue;

        const cv::Mat descriptorMP = pMP->GetDescriptor();
        int bestDist = INT_MAX;
        int bestIdx = -1;

        for (size_t idx : candidates) {
            if (CurrentFrame.mvpMapPoints[idx]) continue;

            const cv::Mat &descriptor = CurrentFrame.mDescriptors.row(idx);
            int dist = DescriptorDistance(descriptorMP, descriptor);
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= ORBdist) {
            CurrentFrame.mvpMapPoints[bestIdx] = pMP;
            matchesCount++;

            if (mbCheckOrientation) {
                float rotationDiff = pKF->mvKeys[i].angle - CurrentFrame.mvKeys[bestIdx].angle;
                if (rotationDiff < 0) rotationDiff += 360.0f;
                int bin = static_cast<int>(round(rotationDiff * histogramFactor)) % HISTO_LENGTH;
                rotationHistogram[bin].push_back(bestIdx);
            }
        }
    }

    if (mbCheckOrientation) {
        int topBins[3];
        ComputeThreeMaxima(rotationHistogram, HISTO_LENGTH, topBins[0], topBins[1], topBins[2]);

        for (int i = 0; i < HISTO_LENGTH; ++i) {
            if (i != topBins[0] && i != topBins[1] && i != topBins[2]) {
                for (size_t idx : rotationHistogram[i]) {
                    CurrentFrame.mvpMapPoints[idx] = nullptr;
                    matchesCount--;
                }
            }
        }
    }
    return matchesCount;
}

int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
    const int *ptrA = a.ptr<int32_t>();
    const int *ptrB = b.ptr<int32_t>();
    int distance = 0;

    for(int i = 0; i < 8; ++i, ++ptrA, ++ptrB) {
        unsigned int value = *ptrA ^ *ptrB;
        value = value - ((value >> 1) & 0x55555555);
        value = (value & 0x33333333) + ((value >> 2) & 0x33333333);
        distance += (((value + (value >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return distance;
}
} 