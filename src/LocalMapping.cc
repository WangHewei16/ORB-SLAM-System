#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>
#include <unistd.h>
namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular)
    : mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), 
      mbFinished(true), mpMap(pMap), mbAbortBA(false), mbStopped(false), 
      mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker = pTracker;
}

void LocalMapping::Run()
{
    mbFinished = false;

    while (!CheckFinish())
    {
        SetAcceptKeyFrames(false);

        if (ProcessKeyFrames())
            continue;

        if (Stop())
        {
            WaitForStopOrFinish();
            if (CheckFinish())
                break;
        }

        ResetIfRequested();
        SetAcceptKeyFrames(true);

        usleep(3000);
    }

    SetFinish();
}

bool LocalMapping::ProcessKeyFrames()
{
    if (!CheckNewKeyFrames())
        return false;

    do {
        ProcessNewKeyFrame();
        MapPointCulling();
        CreateNewMapPoints();

        if (mbAbortBA || stopRequested() || CheckNewKeyFrames())
            continue;

        PerformLocalBundleAdjustment();

    } while (CheckNewKeyFrames());

    mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);

    return true;
}

void LocalMapping::PerformLocalBundleAdjustment()
{
    if(mpMap->KeyFramesInMap() > 2)
        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);
    KeyFrameCulling();
}

void LocalMapping::WaitForStopOrFinish()
{
    while (isStopped() && !CheckFinish())
    {
        usleep(3000);
    }
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA = true;
}

bool LocalMapping::CheckNewKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexNewKFs);
    return (!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    std::unique_lock<std::mutex> lock(mMutexNewKFs);
    mpCurrentKeyFrame = mlNewKeyFrames.front();
    mlNewKeyFrames.pop_front();

    mpCurrentKeyFrame->ComputeBoW();
    AssociateAndUpdateMapPoints();
    mpCurrentKeyFrame->UpdateConnections();
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::AssociateAndUpdateMapPoints() {
    const std::vector<MapPoint*>& vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for (size_t i = 0; i < vpMapPointMatches.size(); i++) {
        MapPoint* pMP = vpMapPointMatches[i];
        if (pMP && !pMP->isBad()) {
            if (!pMP->IsInKeyFrame(mpCurrentKeyFrame)) {
                pMP->AddObservation(mpCurrentKeyFrame, i);
                pMP->UpdateNormalAndDepth();
                pMP->ComputeDistinctiveDescriptors();
            } else {
                mlpRecentAddedMapPoints.push_back(pMP);
            }
        }
    }
}

void LocalMapping::MapPointCulling()
{
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;
    int nThObs = mbMonocular ? 2 : 3;
    const float minFoundRatio = 0.25f;

    for (auto lit = mlpRecentAddedMapPoints.begin(); lit != mlpRecentAddedMapPoints.end(); )
    {
        MapPoint* pMP = *lit;
        if (pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else
        {
            const int age = nCurrentKFid - pMP->mnFirstKFid;
            const bool cullByObs = (age >= 2 && pMP->Observations() <= nThObs);
            const bool cullByFoundRatio = (pMP->GetFoundRatio() < minFoundRatio);

            if (cullByObs || cullByFoundRatio)
            {
                pMP->SetBadFlag();
                lit = mlpRecentAddedMapPoints.erase(lit);
            }
            else
            {
                ++lit;
            }
        }
    }
}

void LocalMapping::CreateNewMapPoints()
{
    int nn = mbMonocular ? 20 : 10;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6, false);
    auto Tcw1 = ComputeCameraTransform(mpCurrentKeyFrame);
    auto Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    int nnew = 0;

    for (KeyFrame* pKF2 : vpNeighKFs)
    {
        if (nnew > 0 && CheckNewKeyFrames())
            return;

        if (!IsBaselineTooShort(pKF2, Ow1))
            continue;

        auto F12 = ComputeF12(mpCurrentKeyFrame, pKF2);
        vector<pair<size_t, size_t>> vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false);
        auto Tcw2 = ComputeCameraTransform(pKF2);

        TriangulateMatches(mpCurrentKeyFrame, pKF2, vMatchedIndices, Tcw1, Tcw2, Ow1, nnew);
    }
}

bool LocalMapping::IsBaselineTooShort(KeyFrame* pKF2, const cv::Mat& Ow1) const
{
    cv::Mat Ow2 = pKF2->GetCameraCenter();
    cv::Mat vBaseline = Ow2 - Ow1;
    float baseline = cv::norm(vBaseline);

    if (!mbMonocular)
        return baseline >= pKF2->mb;
    else
    {
        float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
        return baseline / medianDepthKF2 >= 0.01;
    }
}

cv::Mat LocalMapping::ComputeCameraTransform(KeyFrame* pKF) const
{
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();
    cv::Mat Tcw(3, 4, CV_32F);
    Rcw.copyTo(Tcw.colRange(0, 3));
    tcw.copyTo(Tcw.col(3));
    return Tcw;
}

void LocalMapping::TriangulateMatches(KeyFrame* pKF1, KeyFrame* pKF2, const vector<pair<size_t, size_t>>& matchedIndices, const cv::Mat& Tcw1, const cv::Mat& Tcw2, const cv::Mat& Ow1, int& nnew)
{
    for (const auto& match : matchedIndices)
    {
        if (TriangulateAndAddPoint(pKF1, pKF2, match, Tcw1, Tcw2, Ow1))
            nnew++;
    }
}

bool LocalMapping::TriangulateAndAddPoint(KeyFrame* pKF1, KeyFrame* pKF2, const pair<size_t, size_t>& match, const cv::Mat& Tcw1, const cv::Mat& Tcw2, const cv::Mat& Ow1)
{
    const int idx1 = match.first;
    const int idx2 = match.second;
    return true;
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = mbMonocular ? 20 : 10;
    const auto vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;

    // Populate target keyframes and extend to second neighbors
    for (auto* pKFi : vpNeighKFs) {
        if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId) {
            continue;
        }
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        auto vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for (auto* pKFi2 : vpSecondNeighKFs) {
            if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId || pKFi2->mnId == mpCurrentKeyFrame->mnId) {
                continue;
            }
            vpTargetKFs.push_back(pKFi2);
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    auto vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (auto* pKFi : vpTargetKFs) {
        matcher.Fuse(pKFi, vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

    for (auto* pKFi : vpTargetKFs) {
        auto vpMapPointsKFi = pKFi->GetMapPointMatches();
        for (auto* pMP : vpMapPointsKFi) {
            if (!pMP || pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId) {
                continue;
            }
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (auto* pMP : vpMapPointMatches) {
        if (pMP && !pMP->isBad()) {
            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();
        }
    }

    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame*& pKF1, KeyFrame*& pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat R2w = pKF2->GetRotation().t();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat t12 = -R1w * R2w * t2w + pKF1->GetTranslation();
    cv::Mat t12x = SkewSymmetricMatrix(t12);
    return pKF1->mK.t().inv() * t12x * R1w * R2w * pKF2->mK.inv();
}

void LocalMapping::RequestStop()
{
    std::lock(mMutexStop, mMutexNewKFs);
    std::lock_guard<std::mutex> lk1(mMutexStop, std::adopt_lock);
    std::lock_guard<std::mutex> lk2(mMutexNewKFs, std::adopt_lock);
    mbStopRequested = true;
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    std::lock_guard<std::mutex> lock(mMutexStop);
    if (mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        std::cout << "Local Mapping STOP" << std::endl;
        return true;
    }
    return false;
}

bool LocalMapping::isStopped()
{
    std::lock_guard<std::mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    std::lock_guard<std::mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    std::lock(mMutexStop, mMutexFinish);
    std::lock_guard<std::mutex> lk1(mMutexStop, std::adopt_lock);
    std::lock_guard<std::mutex> lk2(mMutexFinish, std::adopt_lock);

    if (mbFinished) return;

    mbStopped = false;
    mbStopRequested = false;
    for (auto* kf : mlNewKeyFrames)
        delete kf;
    mlNewKeyFrames.clear();

    std::cout << "Local Mapping RELEASE" << std::endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    std::lock_guard<std::mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    std::lock_guard<std::mutex> lock(mMutexAccept);
    mbAcceptKeyFrames = flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    std::lock_guard<std::mutex> lock(mMutexStop);
    if (flag && mbStopped)
        return false;

    mbNotStop = flag;
    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling() {
    auto vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();
    for (auto* pKF : vpLocalKeyFrames) {
        if (pKF->mnId == 0) continue;

        auto vpMapPoints = pKF->GetMapPointMatches();
        int nRedundantObservations = 0;
        int nMPs = 0;
        int thresholdObs = 3;

        for (size_t i = 0; i < vpMapPoints.size(); i++) {
            MapPoint* pMP = vpMapPoints[i];
            if (pMP && !pMP->isBad() && (!mbMonocular || (pKF->mvDepth[i] > pKF->mThDepth && pKF->mvDepth[i] >= 0))) {
                nMPs++;
                if (pMP->Observations() > thresholdObs) {
                    int nObs = std::count_if(pMP->GetObservations().begin(), pMP->GetObservations().end(), 
                                             [pKF, &pMP, i](const std::pair<KeyFrame*, size_t>& obs) {
                                                 return obs.first != pKF && obs.first->mvKeysUn[obs.second].octave <= pKF->mvKeysUn[i].octave + 1;
                                             });
                    if (nObs >= thresholdObs) nRedundantObservations++;
                }
            }
        }

        if (nRedundantObservations > 0.9 * nMPs) {
            pKF->SetBadFlag();
        }
    }
}
cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v) {
    return (cv::Mat_<float>(3,3) << 0, -v.at<float>(2), v.at<float>(1),
                                    v.at<float>(2), 0, -v.at<float>(0),
                                    -v.at<float>(1), v.at<float>(0), 0);
}

void LocalMapping::RequestReset() {
    {
        std::unique_lock<std::mutex> lock(mMutexReset);
        mbResetRequested = true;
    }
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        std::unique_lock<std::mutex> lock(mMutexReset);
        if (!mbResetRequested) break;
    }
}

void LocalMapping::ResetIfRequested() {
    std::unique_lock<std::mutex> lock(mMutexReset);
    if (mbResetRequested) {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested = false;
    }
}

void LocalMapping::RequestFinish() {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish() {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish() {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinished = true;
    std::unique_lock<std::mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished() {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}
}