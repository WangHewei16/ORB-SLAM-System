#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>
#include <unistd.h>

using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor)
    : mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    mMinFrames = 0;
    mMaxFrames = fps;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp) {
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if (mImGray.channels() > 1) {
        if (mbRGB) {
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
        } else {
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
    Track();
    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp) {
    mImGray = imRGB;

    if (mImGray.channels() > 1) {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }

    cv::Mat imDepth = imD;
    if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
        imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

    mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
    Track();
    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp) {
    mImGray = im;

    if (mImGray.channels() > 1) {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }

    if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);
    else
        mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

    Track();
    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track() {
    if (mState == NO_IMAGES_YET) {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState = mState;

    // Lock Map for this scope, ensuring no changes
    std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

    if (mState == NOT_INITIALIZED) {
        MonocularInitialization();  // Handles both Mono and Stereo/RGBD cases inside

        mpFrameDrawer->Update(this);

        if (mState != OK) return;
    }

    bool bOK = false;

    // Decide tracking strategy: motion model or relocalization
    if (mState == OK) {
        CheckReplacedInLastFrame();
        bOK = !mVelocity.empty() && mCurrentFrame.mnId >= mnLastRelocFrameId + 2 ?
              TrackWithMotionModel() : TrackReferenceKeyFrame();
    } else {
        bOK = Relocalization();
    }

    // Tracking either by motion, reference keyframe or by relocalization
    if (!bOK) bOK = TrackReferenceKeyFrame();
    if (!bOK) bOK = Relocalization();

    mCurrentFrame.mpReferenceKF = mpReferenceKF;

    // Local map tracking
    if (bOK) bOK = TrackLocalMap();

    mState = bOK ? OK : LOST;

    mpFrameDrawer->Update(this);

    if (bOK) {
        UpdateMotionModel();

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        CleanVOMatches();

        if (NeedNewKeyFrame()) CreateNewKeyFrame();

        // Remove outlier observations
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                mCurrentFrame.mvpMapPoints[i] = nullptr;
        }
    }

    if (mState == LOST && mpMap->KeyFramesInMap() <= 5) {
        std::cout << "Track lost soon after initialisation, resetting..." << std::endl;
        mpSystem->Reset();
        return;
    }

    mLastFrame = Frame(mCurrentFrame);
    StoreFrameInformation();
}

void Tracking::StoreFrameInformation() {
    if (!mCurrentFrame.mTcw.empty()) {
        cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState == LOST);
    } else {
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState == LOST);
    }
}

void Tracking::UpdateMotionModel() {
    if (!mLastFrame.mTcw.empty()) {
        cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
        mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
        mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
        mVelocity = mCurrentFrame.mTcw * LastTwc;
    } else {
        mVelocity = cv::Mat();
    }
}

void Tracking::CleanVOMatches() {
    for (int i = 0; i < mCurrentFrame.N; i++) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP && pMP->Observations() < 1) {
            mCurrentFrame.mvbOutlier[i] = false;
            mCurrentFrame.mvpMapPoints[i] = nullptr;
        }
    }
}

void Tracking::StereoInitialization() {
    if (mCurrentFrame.N > 500) {
        mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);
        mpMap->AddKeyFrame(pKFini);

        for (int i = 0; i < mCurrentFrame.N; i++) {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0) {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);
                pNewMP->AddObservation(pKFini, i);
                pKFini->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);
                mCurrentFrame.mvpMapPoints[i] = pNewMP;
            }
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mState = OK;
        mLastFrame = mCurrentFrame;
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;
        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;
    }
}

void Tracking::MonocularInitialization() {
    if (!mpInitializer) {
        if (mCurrentFrame.mvKeys.size() > 100) {
            mInitialFrame = mCurrentFrame;
            mLastFrame = mCurrentFrame;
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            std::transform(mCurrentFrame.mvKeysUn.begin(), mCurrentFrame.mvKeysUn.end(), mvbPrevMatched.begin(),
                           [](const cv::KeyPoint& kp) { return kp.pt; });

            delete mpInitializer;  // Safely delete the initializer if it exists
            mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);
            mvIniMatches.assign(mCurrentFrame.mvKeys.size(), -1);
            return;
        }
    } else {
        if (mCurrentFrame.mvKeys.size() <= 100) {
            delete mpInitializer;
            mpInitializer = nullptr;
            mvIniMatches.assign(mvIniMatches.size(), -1);
            return;
        }

        ORBmatcher matcher(0.9, true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);
        if (nmatches < 100) {
            delete mpInitializer;
            mpInitializer = nullptr;
            return;
        }
        cv::Mat Rcw, tcw;
        vector<bool> vbTriangulated;
        if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated)) {
            auto filter = [&](size_t i) { return mvIniMatches[i] >= 0 && !vbTriangulated[i]; };
            std::replace_if(mvIniMatches.begin(), mvIniMatches.end(), filter, -1);
            mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular() {
    KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    for (size_t i = 0; i < mvIniMatches.size(); i++) {
        if (mvIniMatches[i] < 0) continue;

        cv::Mat worldPos = cv::Mat(mvIniP3D[i]).reshape(1);

        MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);
        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        mpMap->AddMapPoint(pMP);
    }

    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;

    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
        cout << "Wrong initialization, resetting..." << endl;
        Reset();
        return;
    }

    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0, 3) *= invMedianDepth;
    pKFcur->SetPose(Tc2w);

    std::vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches(); 
    for (MapPoint* pMP : vpAllMapPoints) {
        if (pMP) {
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());
    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState = OK;
}

void Tracking::CheckReplacedInLastFrame() {
    for (int i = 0; i < mLastFrame.N; ++i) {
        auto& pMP = mLastFrame.mvpMapPoints[i];
        if (pMP) {
            MapPoint* pRep = pMP->GetReplaced();
            if (pRep) {
                pMP = pRep;
            }
        }
    }
}

bool Tracking::TrackReferenceKeyFrame() {
    mCurrentFrame.ComputeBoW();

    ORBmatcher matcher(0.7, true);
    vector<MapPoint*> vpMapPointMatches;
    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15) return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; ++i) {
        auto& pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP) {
            if (mCurrentFrame.mvbOutlier[i]) {
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP = nullptr;
                mCurrentFrame.mvbOutlier[i] = false;
                nmatches--;
            } else if (pMP->Observations() > 0) {
                nmatchesMap++;
            }
        }
    }

    return nmatchesMap >= 10;
}

void Tracking::UpdateLastFrame() {
    if (mLastFrame.mpReferenceKF) {
        cv::Mat Tlr = mlRelativeFramePoses.back();
        mLastFrame.SetPose(Tlr * mLastFrame.mpReferenceKF->GetPose());
    }

    if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking)
        return;

    vector<pair<float, int>> vDepthIdx;
    for (int i = 0; i < mLastFrame.N; i++) {
        float z = mLastFrame.mvDepth[i];
        if (z > 0) {
            vDepthIdx.emplace_back(z, i);
        }
    }

    if (vDepthIdx.empty()) return;

    sort(vDepthIdx.begin(), vDepthIdx.end());

    int nPoints = 0;
    for (const auto& [depth, i] : vDepthIdx) {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if (!pMP || pMP->Observations() < 1) {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);
            mLastFrame.mvpMapPoints[i] = pNewMP;
            mlpTemporalPoints.push_back(pNewMP);
        }
        nPoints++;
        if (depth > mThDepth && nPoints > 100) break;
    }
}

bool Tracking::TrackWithMotionModel() {
    ORBmatcher matcher(0.9, true);
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
    std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);

    int th = (mSensor != System::STEREO) ? 15 : 7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

    if (nmatches < 20) {
        std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th, mSensor == System::MONOCULAR);
    }

    if (nmatches < 20) return false;

    Optimizer::PoseOptimization(&mCurrentFrame);

    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++) {
        auto& mp = mCurrentFrame.mvpMapPoints[i];
        if (mp) {
            if (mCurrentFrame.mvbOutlier[i]) {
                mp = nullptr;
                mCurrentFrame.mvbOutlier[i] = false;
                mp->mbTrackInView = false;
                mp->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            } else if (mp->Observations() > 0) {
                nmatchesMap++;
            }
        }
    }

    mbVO = mbOnlyTracking && nmatchesMap < 10;
    return mbOnlyTracking ? nmatches > 20 : nmatchesMap >= 10;
}

bool Tracking::TrackLocalMap() {
    UpdateLocalMap();
    SearchLocalPoints();

    Optimizer::PoseOptimization(&mCurrentFrame);

    int mnMatchesInliers = 0;
    for (int i = 0; i < mCurrentFrame.N; i++) {
        auto& mp = mCurrentFrame.mvpMapPoints[i];
        if (mp) {
            if (!mCurrentFrame.mvbOutlier[i]) {
                mp->IncreaseFound();
                if (mbOnlyTracking || mp->Observations() > 0) {
                    mnMatchesInliers++;
                }
            } else if (mSensor == System::STEREO) {
                mp = nullptr;
            }
        }
    }

    bool recentlyRelocalized = mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames;
    if ((recentlyRelocalized && mnMatchesInliers < 50) || mnMatchesInliers < 30) {
        return false;
    }

    return true;
}

bool Tracking::NeedNewKeyFrame() {
    if(mbOnlyTracking) return false;

    // Conditions that prevent keyframe creation
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) return false;
    const int numKeyFrames = mpMap->KeyFramesInMap();
    if(mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && numKeyFrames > mMaxFrames) return false;

    int requiredObservations = numKeyFrames <= 2 ? 2 : 3;
    int trackedMapPoints = mpReferenceKF->TrackedMapPoints(requiredObservations);
    bool isLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Keyframe conditions based on tracking and map integration quality
    int closeNonTracked = 0, closeTracked = 0;
    if(mSensor != System::MONOCULAR) {
        for(int i = 0; i < mCurrentFrame.N; i++) {
            float depth = mCurrentFrame.mvDepth[i];
            if(depth > 0 && depth < mThDepth) {
                (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i] ? closeTracked : closeNonTracked)++;
            }
        }
    }

    bool needMoreClosePoints = closeTracked < 100 && closeNonTracked > 70;
    float referenceRatioThreshold = numKeyFrames < 2 ? 0.4f : (mSensor == System::MONOCULAR ? 0.9f : 0.75f);

    // Evaluate keyframe conditions
    bool condition1 = mCurrentFrame.mnId >= mnLastKeyFrameId + (needMoreClosePoints ? mMinFrames : mMaxFrames);
    bool condition2 = trackedMapPoints < referenceRatioThreshold * trackedMapPoints || needMoreClosePoints;

    if(condition1 && condition2) {
        if(isLocalMappingIdle) {
            return true;
        } else {
            mpLocalMapper->InterruptBA();
            return mSensor != System::MONOCULAR && mpLocalMapper->KeyframesInQueue() < 3;
        }
    }
    return false;
}

void Tracking::CreateNewKeyFrame() {
    if(!mpLocalMapper->SetNotStop(true)) return;

    KeyFrame* newKeyFrame = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);
    mpReferenceKF = newKeyFrame;
    mCurrentFrame.mpReferenceKF = newKeyFrame;

    if(mSensor != System::MONOCULAR) {
        vector<pair<float, int>> depthIndex;
        for(int i = 0; i < mCurrentFrame.N; i++) {
            float depth = mCurrentFrame.mvDepth[i];
            if(depth > 0) depthIndex.emplace_back(depth, i);
        }

        if(!depthIndex.empty()) {
            sort(depthIndex.begin(), depthIndex.end());
            int count = 0;
            for(auto &depth_pair : depthIndex) {
                int idx = depth_pair.second;
                MapPoint* mapPoint = mCurrentFrame.mvpMapPoints[idx];
                if(!mapPoint || mapPoint->Observations() < 1) {
                    cv::Mat point3D = mCurrentFrame.UnprojectStereo(idx);
                    MapPoint* newMapPoint = new MapPoint(point3D, newKeyFrame, mpMap);
                    newMapPoint->AddObservation(newKeyFrame, idx);
                    newKeyFrame->AddMapPoint(newMapPoint, idx);
                    newMapPoint->ComputeDistinctiveDescriptors();
                    newMapPoint->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(newMapPoint);
                    mCurrentFrame.mvpMapPoints[idx] = newMapPoint;
                }
                count++;
                if(depth_pair.first > mThDepth && count > 100) break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(newKeyFrame);
    mpLocalMapper->SetNotStop(false);
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = newKeyFrame;
}

void Tracking::SearchLocalPoints() {
    // Do not search already matched map points
    for(auto it = mCurrentFrame.mvpMapPoints.begin(); it != mCurrentFrame.mvpMapPoints.end(); ++it) {
        MapPoint* pMP = *it;
        if(pMP) {
            if(pMP->isBad()) {
                *it = nullptr;
            } else {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch = 0;

    // Project points in frame and check visibility
    for(auto it = mvpLocalMapPoints.begin(); it != mvpLocalMapPoints.end(); ++it) {
        MapPoint* pMP = *it;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId || pMP->isBad()) continue;
        // Project and check visibility
        if(mCurrentFrame.isInFrustum(pMP, 0.5)) {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch > 0) {
        ORBmatcher matcher(0.8);
        int th = mSensor == System::RGBD ? 3 : 1;
        if(mCurrentFrame.mnId < mnLastRelocFrameId + 2) th = 5;
        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
    }
}

void Tracking::UpdateLocalMap() {
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints() {
    mvpLocalMapPoints.clear();

    for(auto itKF = mvpLocalKeyFrames.begin(); itKF != mvpLocalKeyFrames.end(); ++itKF) {
        KeyFrame* pKF = *itKF;
        const auto vpMPs = pKF->GetMapPointMatches();

        for(auto itMP = vpMPs.begin(); itMP != vpMPs.end(); ++itMP) {
            MapPoint* pMP = *itMP;
            if(!pMP || pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId || pMP->isBad()) continue;
            mvpLocalMapPoints.push_back(pMP);
            pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    map<KeyFrame*, int> keyframeCounter;
    for (int i = 0; i < mCurrentFrame.N; i++) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP) {
            if (!pMP->isBad()) {
                const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                for (auto it = observations.begin(); it != observations.end(); it++) {
                    keyframeCounter[it->first]++;
                }
            } else {
                mCurrentFrame.mvpMapPoints[i] = nullptr;
            }
        }
    }

    if (keyframeCounter.empty())
        return;

    int max = 0;
    KeyFrame* pKFmax = nullptr;

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // Include all keyframes that observe a map point and also check which keyframe shares most points
    for (auto it = keyframeCounter.begin(); it != keyframeCounter.end(); it++) {
        KeyFrame* pKF = it->first;
        if (!pKF->isBad()) {
            if (it->second > max) {
                max = it->second;
                pKFmax = pKF;
            }
            mvpLocalKeyFrames.push_back(pKF);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }
    }

    // Include some additional keyframes that are neighbors to already included keyframes
    for (auto itKF = mvpLocalKeyFrames.begin(); itKF != mvpLocalKeyFrames.end(); ++itKF) {
        if (mvpLocalKeyFrames.size() > 80)
            break;

        KeyFrame* pKF = *itKF;
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for (auto pNeighKF : vNeighs) {
            if (!pNeighKF->isBad() && pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                mvpLocalKeyFrames.push_back(pNeighKF);
                pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }

        const set<KeyFrame*> childs = pKF->GetChilds();
        for (auto pChildKF : childs) {
            if (!pChildKF->isBad() && pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                mvpLocalKeyFrames.push_back(pChildKF);
                pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if (pParent && pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
            mvpLocalKeyFrames.push_back(pParent);
            pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }
    }

    if (pKFmax) {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    mCurrentFrame.ComputeBoW();
    vector<KeyFrame*> candidateKeyFrames = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    if (candidateKeyFrames.empty())
        return false;

    ORBmatcher matcher(0.75, true);
    vector<PnPsolver*> pnpSolvers(candidateKeyFrames.size(), nullptr);
    vector<vector<MapPoint*>> mapPointMatches(candidateKeyFrames.size());
    vector<bool> discarded(candidateKeyFrames.size(), false);

    int validCandidates = 0;

    for (size_t i = 0; i < candidateKeyFrames.size(); ++i) {
        KeyFrame* keyFrame = candidateKeyFrames[i];
        if (keyFrame->isBad()) {
            discarded[i] = true;
            continue;
        }

        int matches = matcher.SearchByBoW(keyFrame, mCurrentFrame, mapPointMatches[i]);
        if (matches < 15) {
            discarded[i] = true;
        } else {
            PnPsolver* solver = new PnPsolver(mCurrentFrame, mapPointMatches[i]);
            solver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
            pnpSolvers[i] = solver;
            validCandidates++;
        }
    }

    bool matchFound = false;
    ORBmatcher finalMatcher(0.9, true);

    while (validCandidates > 0 && !matchFound) {
        for (size_t i = 0; i < candidateKeyFrames.size(); ++i) {
            if (discarded[i]) continue;

            vector<bool> inliers;
            int inliersCount;
            bool noMore;

            cv::Mat Tcw = pnpSolvers[i]->iterate(5, noMore, inliers, inliersCount);
            if (noMore) {
                discarded[i] = true;
                validCandidates--;
                continue;
            }

            if (!Tcw.empty()) {
                mCurrentFrame.SetPose(Tcw);
                set<MapPoint*> foundMapPoints;

                for (size_t j = 0; j < inliers.size(); ++j) {
                    if (inliers[j]) {
                        mCurrentFrame.mvpMapPoints[j] = mapPointMatches[i][j];
                        foundMapPoints.insert(mapPointMatches[i][j]);
                    } else {
                        mCurrentFrame.mvpMapPoints[j] = nullptr;
                    }
                }

                int goodMatches = Optimizer::PoseOptimization(&mCurrentFrame);

                if (goodMatches < 10) continue;

                int additionalMatches = finalMatcher.SearchByProjection(mCurrentFrame, candidateKeyFrames[i], foundMapPoints, 10, 100);
                goodMatches += additionalMatches;

                if (goodMatches >= 50) {
                    matchFound = true;
                    break;
                }
            }
        }
    }

    for (PnPsolver* solver : pnpSolvers) {
        if (solver) delete solver;
    }

    if (!matchFound) {
        return false;
    } else {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}


void Tracking::Reset()
{
    cout << "System Reseting" << endl;
    
    if (mpViewer) {
        mpViewer->RequestStop();
        while (!mpViewer->isStopped()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
    }

    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    delete mpInitializer;
    mpInitializer = nullptr;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if (mpViewer) {
        mpViewer->Release();
    }
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    float k3 = fSettings["Camera.k3"];
    if(k3 != 0.0f) // Properly check against float zero after directly retrieving the value.
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}
}