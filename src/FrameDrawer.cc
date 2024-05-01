#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <mutex>
#include <unistd.h>

namespace ORB_SLAM2 {

    FrameDrawer::FrameDrawer(Map* pMap) : mpMap(pMap) {
        mState = Tracking::SYSTEM_NOT_READY;
        mIm = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    cv::Mat FrameDrawer::DrawFrame() {
        cv::Mat frameImage;
        std::vector<cv::KeyPoint> initialKeypoints, currentKeypoints;
        std::vector<int> keypointMatches;
        std::vector<bool> trackingPointsVO, trackingPointsMap;
        int trackingState;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            trackingState = mState;
            if (mState == Tracking::SYSTEM_NOT_READY)
                mState = Tracking::NO_IMAGES_YET;
            mIm.copyTo(frameImage);
            if (mState == Tracking::NOT_INITIALIZED) {
                initialKeypoints = mvIniKeys;
                currentKeypoints = mvCurrentKeys;
                keypointMatches = mvIniMatches;
            } else if (mState == Tracking::OK) {
                currentKeypoints = mvCurrentKeys;
                trackingPointsVO = mvbVO;
                trackingPointsMap = mvbMap;
            } else if (mState == Tracking::LOST) {
                currentKeypoints = mvCurrentKeys;
            }
        }
        if (frameImage.channels() < 3)
            cv::cvtColor(frameImage, frameImage, cv::COLOR_GRAY2BGR);
        if (trackingState == Tracking::NOT_INITIALIZED) {
            for (size_t i = 0; i < keypointMatches.size(); i++) {
                if (keypointMatches[i] >= 0) {
                    cv::line(frameImage, initialKeypoints[i].pt, currentKeypoints[keypointMatches[i]].pt, cv::Scalar(0, 255, 0));
                }
            }
        } else if (trackingState == Tracking::OK) {
            int trackedCount = 0, trackedVOCount = 0;
            const float radius = 5.0;
            for (size_t i = 0; i < currentKeypoints.size(); i++) {
                if (trackingPointsVO[i] || trackingPointsMap[i]) {
                    cv::Point2f topLeft = currentKeypoints[i].pt - cv::Point2f(radius, radius);
                    cv::Point2f bottomRight = currentKeypoints[i].pt + cv::Point2f(radius, radius);

                    if (trackingPointsMap[i]) {
                        cv::rectangle(frameImage, topLeft, bottomRight, cv::Scalar(0, 255, 0));
                        cv::circle(frameImage, currentKeypoints[i].pt, 2, cv::Scalar(0, 255, 0), -1);
                        trackedCount++;
                    } else {
                        cv::rectangle(frameImage, topLeft, bottomRight, cv::Scalar(255, 0, 0));
                        cv::circle(frameImage, currentKeypoints[i].pt, 2, cv::Scalar(255, 0, 0), -1);
                        trackedVOCount++;
                    }
                }
            }
        }
        cv::Mat annotatedFrame;
        DrawTextInfo(frameImage, trackingState, annotatedFrame);
        return annotatedFrame;
    }

    void FrameDrawer::DrawTextInfo(cv::Mat &frame, int state, cv::Mat &infoFrame) {
        std::ostringstream infoText;
        switch (state) {
            case Tracking::NO_IMAGES_YET:
                infoText << " WAITING FOR IMAGES";
                break;
            case Tracking::NOT_INITIALIZED:
                infoText << " TRYING TO INITIALIZE ";
                break;
            case Tracking::OK:
                infoText << (mbOnlyTracking ? "LOCALIZATION | " : "SLAM MODE | ");
                infoText << "KFs: " << mpMap->KeyFramesInMap() << ", MPs: " << mpMap->MapPointsInMap() << ", Matches: " << mnTracked;
                if (mnTrackedVO > 0)
                    infoText << ", + VO matches: " << mnTrackedVO;
                break;
            case Tracking::LOST:
                infoText << " TRACK LOST. TRYING TO RELOCALIZE ";
                break;
            case Tracking::SYSTEM_NOT_READY:
                infoText << " LOADING ORB VOCABULARY. PLEASE WAIT...";
                break;
        }

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(infoText.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

        infoFrame = cv::Mat(frame.rows + textSize.height + 10, frame.cols, frame.type());
        frame.copyTo(infoFrame.rowRange(0, frame.rows).colRange(0, frame.cols));
        infoFrame.rowRange(frame.rows, infoFrame.rows) = cv::Mat::zeros(textSize.height + 10, frame.cols, frame.type());
        cv::putText(infoFrame, infoText.str(), cv::Point(5, infoFrame.rows - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
    }

    void FrameDrawer::Update(Tracking *pTracker) {
        std::lock_guard<std::mutex> lock(mMutex);
        pTracker->mImGray.copyTo(mIm);
        mvCurrentKeys = pTracker->mCurrentFrame.mvKeys;
        size_t keyCount = mvCurrentKeys.size();
        mvbVO.resize(keyCount, false);
        mvbMap.resize(keyCount, false);
        mbOnlyTracking = pTracker->mbOnlyTracking;

        if (pTracker->mLastProcessedState == Tracking::NOT_INITIALIZED) {
            mvIniKeys = pTracker->mInitialFrame.mvKeys;
            mvIniMatches = pTracker->mvIniMatches;
        } else if (pTracker->mLastProcessedState == Tracking::OK) {
            size_t i = 0;
            while (i < keyCount) {
                MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
                if (pMP && !pTracker->mCurrentFrame.mvbOutlier[i]) {
                    if (pMP->Observations() > 0) {
                        mvbMap[i] = true;
                    } else {
                        mvbVO[i] = true;
                    }
                }
                i++;
            }
        }
        mState = static_cast<int>(pTracker->mLastProcessedState);
    }
}

