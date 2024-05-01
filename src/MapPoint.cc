#include "MapPoint.h"
#include "ORBmatcher.h"

#include <mutex>
#include <unistd.h>

namespace ORB_SLAM2
{

    long unsigned int MapPoint::nNextId = 0;
    std::mutex MapPoint::mGlobalMutex;

// Constructor for map point from keyframe
    MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap) :
            mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0),
            mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0),
            mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
            mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1),
            mnFound(1), mbBad(false), mpReplaced(nullptr), mfMinDistance(0),
            mfMaxDistance(0), mpMap(pMap)
    {
        Pos.copyTo(mWorldPos);
        mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

        std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

// Constructor for map point from frame
    MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF) :
            mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0),
            mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0),
            mnLoopPointForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0),
            mnBAGlobalForKF(0), mpRefKF(nullptr), mnVisible(1), mnFound(1),
            mbBad(false), mpReplaced(nullptr), mpMap(pMap)
    {
        Pos.copyTo(mWorldPos);
        cv::Mat cameraCenter = pFrame->GetCameraCenter();
        mNormalVector = mWorldPos - cameraCenter;
        mNormalVector /= cv::norm(mNormalVector);

        cv::Mat PC = Pos - cameraCenter;
        float distance = cv::norm(PC);
        int level = pFrame->mvKeysUn[idxF].octave;
        float scaleFactor = pFrame->mvScaleFactors[level];
        int numLevels = pFrame->mnScaleLevels;
        mfMaxDistance = distance * scaleFactor;
        mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[numLevels - 1];
        pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);
        std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    void MapPoint::SetWorldPos(const cv::Mat &Pos)
    {
        std::unique_lock<std::mutex> lock1(mGlobalMutex);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        Pos.copyTo(mWorldPos);
    }

    cv::Mat MapPoint::GetWorldPos()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return mWorldPos.clone();
    }

    cv::Mat MapPoint::GetNormal()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return mNormalVector.clone();
    }

    KeyFrame* MapPoint::GetReferenceKeyFrame()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if (mObservations.find(pKF) != mObservations.end())
            return;
        mObservations[pKF] = idx;

        if (pKF->mvuRight[idx] >= 0)
            nObs += 2;
        else
            nObs++;
    }

    void MapPoint::EraseObservation(KeyFrame* pKF)
    {
        bool setBad = false;
        {
            std::unique_lock<std::mutex> lock(mMutexFeatures);
            if (mObservations.find(pKF) != mObservations.end())
            {
                int idx = mObservations[pKF];
                nObs -= (pKF->mvuRight[idx] >= 0) ? 2 : 1;
                mObservations.erase(pKF);

                if (mpRefKF == pKF && !mObservations.empty())
                    mpRefKF = mObservations.begin()->first;

                if (nObs <= 2)
                    setBad = true;
            }
        }

        if (setBad)
            SetBadFlag();
    }

    std::map<KeyFrame*, size_t> MapPoint::GetObservations()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mObservations;
    }

    int MapPoint::Observations()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return nObs;
    }

    void MapPoint::SetBadFlag()
    {
        std::map<KeyFrame*, size_t> tempObservations;
        {
            std::unique_lock<std::mutex> lock1(mMutexFeatures);
            std::unique_lock<std::mutex> lock2(mMutexPos);
            mbBad = true;
            tempObservations = mObservations;
            mObservations.clear();
        }
        for (auto &obs : tempObservations)
        {
            KeyFrame* pKF = obs.first;
            pKF->EraseMapPointMatch(obs.second);
        }

        mpMap->EraseMapPoint(this);
    }

    MapPoint* MapPoint::GetReplaced()
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    void MapPoint::Replace(MapPoint* pMP)
    {
        if (pMP->mnId == this->mnId)
            return;

        int visible, found;
        std::map<KeyFrame*, size_t> observations;
        {
            std::unique_lock<std::mutex> lock1(mMutexFeatures);
            std::unique_lock<std::mutex> lock2(mMutexPos);
            observations = mObservations;
            mObservations.clear();
            mbBad = true;
            visible = mnVisible;
            found = mnFound;
            mpReplaced = pMP;
        }

        for (auto &obs : observations)
        {
            KeyFrame* pKF = obs.first;

            if (!pMP->IsInKeyFrame(pKF))
            {
                pKF->ReplaceMapPointMatch(obs.second, pMP);
                pMP->AddObservation(pKF, obs.second);
            }
            else
            {
                pKF->EraseMapPointMatch(obs.second);
            }
        }
        pMP->IncreaseFound(found);
        pMP->IncreaseVisible(visible);
        pMP->ComputeDistinctiveDescriptors();

        mpMap->EraseMapPoint(this);
    }

    // Computes if a map point should be considered for further processing
    bool MapPoint::isBad()
    {
        std::unique_lock<std::mutex> lockFeatures(mMutexFeatures);
        std::unique_lock<std::mutex> lockPosition(mMutexPos);
        return mbBad;
    }

// Increments the counter for the number of times a point was seen
    void MapPoint::IncreaseVisible(int n)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

// Increments the counter for the number of times a point was found
    void MapPoint::IncreaseFound(int n)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mnFound += n;
    }

// Calculates the ratio of found vs visible
    float MapPoint::GetFoundRatio()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound) / mnVisible;
    }

// Determines the best descriptor based on minimum median distance to others
    void MapPoint::ComputeDistinctiveDescriptors()
    {
        std::vector<cv::Mat> descriptors;
        std::map<KeyFrame*, size_t> localObservations;

        {
            std::unique_lock<std::mutex> lock(mMutexFeatures);
            if (mbBad) return;
            localObservations = mObservations;
        }

        if (localObservations.empty()) return;

        descriptors.reserve(localObservations.size());
        for (const auto& obs : localObservations)
        {
            KeyFrame* keyframe = obs.first;
            if (!keyframe->isBad())
                descriptors.push_back(keyframe->mDescriptors.row(obs.second));
        }

        if (descriptors.empty()) return;

        // Calculate pairwise descriptor distances
        size_t N = descriptors.size();
        std::vector<std::vector<float>> distances(N, std::vector<float>(N, 0));
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = i + 1; j < N; j++)
            {
                int dist = ORBmatcher::DescriptorDistance(descriptors[i], descriptors[j]);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        // Find the descriptor with the minimum median distance
        int bestMedian = INT_MAX, bestIndex = 0;
        for (size_t i = 0; i < N; i++)
        {
            std::vector<int> dists(distances[i].begin(), distances[i].end());
            std::nth_element(dists.begin(), dists.begin() + dists.size()/2, dists.end());
            int median = dists[dists.size() / 2];

            if (median < bestMedian)
            {
                bestMedian = median;
                bestIndex = i;
            }
        }

        // Set the best descriptor as the map point's descriptor
        {
            std::unique_lock<std::mutex> lock(mMutexFeatures);
            mDescriptor = descriptors[bestIndex].clone();
        }
    }

    cv::Mat MapPoint::GetDescriptor()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mDescriptor.clone();
    }

    int MapPoint::GetIndexInKeyFrame(KeyFrame* keyFrame)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if (mObservations.find(keyFrame) != mObservations.end())
            return mObservations[keyFrame];
        else
            return -1;
    }

    bool MapPoint::IsInKeyFrame(KeyFrame* keyFrame)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mObservations.count(keyFrame) > 0;
    }

    void MapPoint::UpdateNormalAndDepth()
    {
        cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
        int count = 0;
        KeyFrame* refKeyFrame;
        cv::Mat position;

        {
            std::unique_lock<std::mutex> lock1(mMutexFeatures);
            std::unique_lock<std::mutex> lock2(mMutexPos);
            if (mbBad) return;
            refKeyFrame = mpRefKF;
            position = mWorldPos.clone();
        }

        if (mObservations.empty()) return;

        for (auto& obs : mObservations)
        {
            KeyFrame* kf = obs.first;
            cv::Mat kfCenter = kf->GetCameraCenter();
            cv::Mat normalComponent = position - kfCenter;
            normal += normalComponent / cv::norm(normalComponent);
            count++;
        }

        {
            std::unique_lock<std::mutex> lock(mMutexPos);
            mNormalVector = normal / count;
            cv::Mat posDiff = position - refKeyFrame->GetCameraCenter();
            float distance = cv::norm(posDiff);
            int level = refKeyFrame->mvKeysUn[mObservations[refKeyFrame]].octave;
            float scaleFactor = refKeyFrame->mvScaleFactors[level];
            int totalLevels = refKeyFrame->mnScaleLevels;

            mfMaxDistance = distance * scaleFactor;
            mfMinDistance = mfMaxDistance / refKeyFrame->mvScaleFactors[totalLevels - 1];
        }
    }

    float MapPoint::GetMinDistanceInvariance()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return 0.8f * mfMinDistance;
    }

    float MapPoint::GetMaxDistanceInvariance()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return 1.2f * mfMaxDistance;
    }

    int MapPoint::PredictScale(const float &currentDist, KeyFrame *keyFrame)
    {
        float ratio;
        {
            std::unique_lock<std::mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }
        int scaleLevel = std::ceil(std::log(ratio) / keyFrame->mfLogScaleFactor);
        return std::max(0, std::min(scaleLevel, keyFrame->mnScaleLevels - 1));
    }

    int MapPoint::PredictScale(const float &currentDist, Frame *frame)
    {
        float ratio;
        {
            std::unique_lock<std::mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }
        int scaleLevel = std::ceil(std::log(ratio) / frame->mfLogScaleFactor);
        return std::max(0, std::min(scaleLevel, frame->mnScaleLevels - 1));
    }
}

