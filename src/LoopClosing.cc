#include "LoopClosing.h"
#include "Sim3Solver.h"
#include "Converter.h"
#include "Optimizer.h"
#include "ORBmatcher.h"
#include<mutex>
#include<thread>
#include <unistd.h>

namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc, bool bFixScale)
    : mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
      mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(nullptr), mLastLoopKFid(0),
      mbRunningGBA(false), mbFinishedGBA(true), mbStopGBA(false), mpThreadGBA(nullptr),
      mbFixScale(bFixScale), mnFullBAIdx(0), mnCovisibilityConsistencyTh(3) {
}

void LoopClosing::SetTracker(Tracking* pTracker) {
    mpTracker = pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping* pLocalMapper) {
    mpLocalMapper = pLocalMapper;
}

void LoopClosing::Run() {
    mbFinished = false;

    while (!CheckFinish()) {
        if (CheckNewKeyFrames() && DetectLoop() && ComputeSim3()) {
            CorrectLoop();
        }

        ResetIfRequested();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame* pKF) {
    std::unique_lock<std::mutex> lock(mMutexLoopQueue);
    if (pKF->mnId != 0) {
        mlpLoopKeyFrameQueue.push_back(pKF);
    }
}

bool LoopClosing::CheckNewKeyFrames() {
    std::unique_lock<std::mutex> lock(mMutexLoopQueue);
    return !mlpLoopKeyFrameQueue.empty();
}

bool LoopClosing::DetectLoop() {
    std::unique_lock<std::mutex> lock(mMutexLoopQueue);
    mpCurrentKF = mlpLoopKeyFrameQueue.front();
    mlpLoopKeyFrameQueue.pop_front();
    mpCurrentKF->SetNotErase();

    if (mpCurrentKF->mnId < mLastLoopKFid + 10) {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    auto vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const auto& CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = std::accumulate(vpConnectedKeyFrames.begin(), vpConnectedKeyFrames.end(), 1.0f,
                                     [&CurrentBowVec, this](float min_score, KeyFrame* pKF) {
                                         if (pKF->isBad()) return min_score;
                                         float score = mpORBVocabulary->score(CurrentBowVec, pKF->mBowVec);
                                         return std::min(min_score, score);
                                     });

    auto vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);
    if (vpCandidateKFs.empty()) {
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    std::vector<ConsistentGroup> vCurrentConsistentGroups;
    std::vector<bool> vbConsistentGroup(mvConsistentGroups.size(), false);

    for (KeyFrame* pCandidateKF : vpCandidateKFs) {
        auto spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;

        for (size_t i = 0; i < mvConsistentGroups.size(); ++i) {
            const auto& [sPreviousGroup, nPreviousConsistency] = mvConsistentGroups[i];
            auto intersect = std::any_of(spCandidateGroup.begin(), spCandidateGroup.end(),
                                         [&sPreviousGroup](KeyFrame* kf) { return sPreviousGroup.count(kf) > 0; });

            if (intersect) {
                int nCurrentConsistency = nPreviousConsistency + 1;
                if (!vbConsistentGroup[i]) {
                    vCurrentConsistentGroups.emplace_back(spCandidateGroup, nCurrentConsistency);
                    vbConsistentGroup[i] = true;
                }
                if (nCurrentConsistency >= mnCovisibilityConsistencyTh) {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent = true;
                }
            }
        }

        if (!bEnoughConsistent) {
            vCurrentConsistentGroups.emplace_back(spCandidateGroup, 0);
        }
    }

    mvConsistentGroups = std::move(vCurrentConsistentGroups);
    mpKeyFrameDB->add(mpCurrentKF);

    if (mvpEnoughConsistentCandidates.empty()) {
        mpCurrentKF->SetErase();
        return false;
    }

    return true;
}

bool LoopClosing::ComputeSim3() {
    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();
    ORBmatcher matcher(0.75, true);
    std::vector<Sim3Solver*> vpSim3Solvers(nInitialCandidates);
    std::vector<std::vector<MapPoint*>> vvpMapPointMatches(nInitialCandidates);
    std::vector<bool> vbDiscarded(nInitialCandidates, false);

    int nCandidates = 0;

    // We assume that pKF needs to be declared outside the loop to maintain scope
    KeyFrame* pKF = nullptr; // Ensure pKF is declared at a function scope

    for (int i = 0; i < nInitialCandidates; i++) {
        pKF = mvpEnoughConsistentCandidates[i]; // Assign the candidate to pKF

        pKF->SetNotErase();

        if (pKF->isBad()) {
            vbDiscarded[i] = true;
            continue;
        }

        int nmatches = matcher.SearchByBoW(mpCurrentKF, pKF, vvpMapPointMatches[i]);

        if (nmatches < 20) {
            vbDiscarded[i] = true;
            continue;
        } else {
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF, pKF, vvpMapPointMatches[i], mbFixScale);
            pSolver->SetRansacParameters(0.99, 20, 300);
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;
    }

    bool bMatch = false;

    // Iterate through candidates to perform RANSAC
    while (nCandidates > 0 && !bMatch) {
        for (int i = 0; i < nInitialCandidates; i++) {
            if (vbDiscarded[i]) continue;

            // Reassign pKF for clarity in the loop, based on consistent candidates
            pKF = mvpEnoughConsistentCandidates[i];

            Sim3Solver* pSolver = vpSim3Solvers[i];
            std::vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            cv::Mat Scm = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

            if (bNoMore) {
                vbDiscarded[i] = true;
                nCandidates--;
                continue;
            }

            if (!Scm.empty()) {
                std::vector<MapPoint*> vpMapPointMatches = filterMatchesByInliers(vvpMapPointMatches[i], vbInliers);
                matcher.SearchBySim3(mpCurrentKF, pKF, vpMapPointMatches, pSolver->GetEstimatedScale(), pSolver->GetEstimatedRotation(), pSolver->GetEstimatedTranslation(), 7.5);

                g2o::Sim3 gScm(Converter::toMatrix3d(pSolver->GetEstimatedRotation()), Converter::toVector3d(pSolver->GetEstimatedTranslation()), pSolver->GetEstimatedScale());
                if (Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale) >= 20) {
                    storeSuccessfulLoop(pKF, gScm, vpMapPointMatches);
                    bMatch = true;
                    break;
                }
            }
        }
    }

    for (int i = 0; i < nInitialCandidates; i++) {
        if (!bMatch || mvpEnoughConsistentCandidates[i] != mpMatchedKF) {
            mvpEnoughConsistentCandidates[i]->SetErase();
        }
    }
    mpCurrentKF->SetErase();

    return bMatch;
}

std::vector<MapPoint*> LoopClosing::filterMatchesByInliers(const std::vector<MapPoint*>& matches, const std::vector<bool>& inliers) {
    std::vector<MapPoint*> filtered;
    for (size_t j = 0; j < inliers.size(); ++j) {
        if (inliers[j]) filtered.push_back(matches[j]);
    }
    return filtered;
}

void LoopClosing::storeSuccessfulLoop(KeyFrame* pKF, const g2o::Sim3& gScm, const std::vector<MapPoint*>& matches) {
    mpMatchedKF = pKF;
    mg2oScw = gScm * g2o::Sim3(Converter::toMatrix3d(pKF->GetRotation()), Converter::toVector3d(pKF->GetTranslation()), 1.0);
    mScw = Converter::toCvMat(mg2oScw);
    mvpCurrentMatchedPoints = matches;
}

void LoopClosing::CorrectLoop() {
    std::cout << "Loop detected!" << std::endl;
    mpLocalMapper->RequestStop();

    if (isRunningGBA()) {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        mbStopGBA = true;
        mnFullBAIdx++;
        if (mpThreadGBA) {
            mpThreadGBA->detach();
            delete mpThreadGBA;
            mpThreadGBA = nullptr;
        }
    }

    while (!mpLocalMapper->isStopped()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    mpCurrentKF->UpdateConnections();
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF] = mg2oScw;
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();

    std::map<KeyFrame*, std::set<KeyFrame*>> LoopConnections;
    std::unique_lock<std::mutex> mapLock(mpMap->mMutexMapUpdate);
    for (KeyFrame* pKFi : mvpCurrentConnectedKFs) {
        cv::Mat Tiw = pKFi->GetPose();
        g2o::Sim3 g2oSiw(Converter::toMatrix3d(Tiw.rowRange(0, 3)), Converter::toVector3d(Tiw.rowRange(0, 3).col(3)), 1.0);
        
        if (pKFi != mpCurrentKF) {
            cv::Mat Tic = Tiw * Twc;
            g2o::Sim3 g2oSic(Converter::toMatrix3d(Tic.rowRange(0, 3)), Converter::toVector3d(Tic.rowRange(0, 3).col(3)), 1.0);
            CorrectedSim3[pKFi] = g2oSic * mg2oScw;
        }
        NonCorrectedSim3[pKFi] = g2oSiw;

        std::set<KeyFrame*> sConnectedKeyframes = pKFi->GetConnectedKeyFrames();
        LoopConnections[pKFi] = sConnectedKeyframes;
    }

    for (auto& [pKFi, g2oCorrectedSiw] : CorrectedSim3) {
        g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();
        g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

        for (MapPoint* pMPi : pKFi->GetMapPointMatches()) {
            if (!pMPi || pMPi->isBad() || pMPi->mnCorrectedByKF == mpCurrentKF->mnId) continue;

            Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(pMPi->GetWorldPos());
            Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));
            pMPi->SetWorldPos(Converter::toCvMat(eigCorrectedP3Dw));
            pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
            pMPi->mnCorrectedReference = pKFi->mnId;
            pMPi->UpdateNormalAndDepth();
        }
        
        Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = g2oCorrectedSiw.translation() / g2oCorrectedSiw.scale();
        cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);
        pKFi->SetPose(correctedTiw);
        pKFi->UpdateConnections();
    }

    SearchAndFuse(CorrectedSim3);
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();
    mpLocalMapper->Release();

    std::cout << "Map updated!" << std::endl;

    mLastLoopKFid = mpCurrentKF->mnId;
}

void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap) {
    ORBmatcher matcher(0.8);
    for (const auto &[pKF, g2oScw] : CorrectedPosesMap) {
        if (!pKF) continue;
        std::vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(), nullptr);
        cv::Mat cvScw = Converter::toCvMat(g2oScw);
        matcher.Fuse(pKF, cvScw, mvpLoopMapPoints, 4, vpReplacePoints);

        std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);
        for (size_t i = 0; i < mvpLoopMapPoints.size(); ++i) {
            if (MapPoint* pRep = vpReplacePoints[i]) {
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}

void LoopClosing::RequestReset() {
    {
        std::unique_lock<std::mutex> lock(mMutexReset);
        mbResetRequested = true;
    }
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::unique_lock<std::mutex> lock(mMutexReset);
        if (!mbResetRequested) break;
    }
}

void LoopClosing::ResetIfRequested() {
    std::unique_lock<std::mutex> lock(mMutexReset);
    if (mbResetRequested) {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid = 0;
        mbResetRequested = false;
    }
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    {
        std::unique_lock<std::mutex> lock(mMutexGBA);
    if (mnFullBAIdx != idx) {
        return; 
    }

    if (!mbStopGBA) {
        std::cout << "Global Bundle Adjustment finished" << std::endl;
        std::cout << "Updating map ..." << std::endl;
        mpLocalMapper->RequestStop();
        
        while (!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::unique_lock<std::mutex> mapLock(mpMap->mMutexMapUpdate);
        
        std::list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(), mpMap->mvpKeyFrameOrigins.end());
        while (!lpKFtoCheck.empty()) {
            KeyFrame* pKF = lpKFtoCheck.front();
            lpKFtoCheck.pop_front();

            if (pKF->mnBAGlobalForKF == nLoopKF) {
                continue; 
            }

            cv::Mat Twc = pKF->GetPoseInverse();
            for (KeyFrame* pChild : pKF->GetChilds()) {
                if (pChild->mnBAGlobalForKF != nLoopKF) {
                    cv::Mat Tchildc = pChild->GetPose() * Twc;
                    pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;
                    pChild->mnBAGlobalForKF = nLoopKF;
                    lpKFtoCheck.push_back(pChild);
                }
            }

            pKF->mTcwBefGBA = pKF->GetPose();
            pKF->SetPose(pKF->mTcwGBA);
        }

        for (MapPoint* pMP : mpMap->GetAllMapPoints()) {
            if (pMP->isBad() || pMP->mnBAGlobalForKF == nLoopKF) {
                continue;
            }

            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            if (pRefKF->mnBAGlobalForKF != nLoopKF) {
                continue;
            }

            cv::Mat Xc = pRefKF->mTcwBefGBA.rowRange(0, 3) * pMP->GetWorldPos() + pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);
            cv::Mat Rwc = pRefKF->GetPoseInverse().rowRange(0, 3);
            cv::Mat twc = pRefKF->GetPoseInverse().rowRange(0, 3).col(3);
            pMP->SetWorldPos(Rwc * Xc + twc);
        }

        mpMap->InformNewBigChange();
        mpLocalMapper->Release();
        std::cout << "Map updated!" << std::endl;
    }

    mbFinishedGBA = true;
    mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish() {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish() {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish() {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished() {
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}
}