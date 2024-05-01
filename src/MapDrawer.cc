#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>
#include <unistd.h>

namespace ORB_SLAM2 {

    MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath): mpMap(pMap) {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
        mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
        mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
        mPointSize = fSettings["Viewer.PointSize"];
        mCameraSize = fSettings["Viewer.CameraSize"];
        mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];
    }

    void MapDrawer::DrawMapPoints() {
        const vector<MapPoint*> &allMapPoints = mpMap->GetAllMapPoints();
        const vector<MapPoint*> &referenceMapPoints = mpMap->GetReferenceMapPoints();

        set<MapPoint*> refPointsSet(referenceMapPoints.begin(), referenceMapPoints.end());

        if(allMapPoints.empty())
            return;

        glPointSize(mPointSize);
        glColor3f(0.0, 0.0, 0.0);
        glBegin(GL_POINTS);
        for (auto mp : allMapPoints) {
            if (mp->isBad() || refPointsSet.count(mp))
                continue;
            cv::Mat position = mp->GetWorldPos();
            glVertex3f(position.at<float>(0), position.at<float>(1), position.at<float>(2));
        }
        glEnd();

        glColor3f(1.0, 0.0, 0.0);
        glBegin(GL_POINTS);
        for (auto refMp : refPointsSet) {
            if (refMp->isBad())
                continue;
            cv::Mat position = refMp->GetWorldPos();
            glVertex3f(position.at<float>(0), position.at<float>(1), position.at<float>(2));
        }
        glEnd();
    }

    void MapDrawer::DrawKeyFrames(bool shouldDrawKeyFrames, bool shouldDrawGraph) {
        const float width = mKeyFrameSize;
        const float height = width * 0.75;
        const float depth = width * 0.6;

        auto keyFrames = mpMap->GetAllKeyFrames();

        if (shouldDrawKeyFrames) {
            for (auto keyFrame : keyFrames) {
                cv::Mat cameraPoseInverse = keyFrame->GetPoseInverse().t();
                glPushMatrix();
                glMultMatrixf(cameraPoseInverse.ptr<GLfloat>(0));
                glLineWidth(mKeyFrameLineWidth);
                glColor3f(0.0f, 0.0f, 1.0f);
                glBegin(GL_LINES);
                glVertex3f(0, 0, 0);
                glVertex3f(width, height, depth);
                glVertex3f(0, 0, 0);
                glVertex3f(width, -height, depth);
                glVertex3f(0, 0, 0);
                glVertex3f(-width, -height, depth);
                glVertex3f(0, 0, 0);
                glVertex3f(-width, height, depth);
                glVertex3f(width, height, depth);
                glVertex3f(width, -height, depth);
                glVertex3f(-width, height, depth);
                glVertex3f(-width, -height, depth);
                glVertex3f(-width, height, depth);
                glVertex3f(width, height, depth);
                glVertex3f(-width, -height, depth);
                glVertex3f(width, -height, depth);
                glEnd();
                glPopMatrix();
            }
        }

        if (shouldDrawGraph) {
            glLineWidth(mGraphLineWidth);
            glColor4f(0.0f, 1.0f, 0.0f, 0.6f);
            glBegin(GL_LINES);
            for (auto kf : keyFrames) {
                cv::Mat cameraCenter = kf->GetCameraCenter();
                auto connectedKeyFrames = kf->GetCovisiblesByWeight(100);
                for (auto connectedKF : connectedKeyFrames) {
                    if (connectedKF->mnId < kf->mnId)
                        continue;
                    cv::Mat connectedCenter = connectedKF->GetCameraCenter();
                    glVertex3f(cameraCenter.at<float>(0), cameraCenter.at<float>(1), cameraCenter.at<float>(2));
                    glVertex3f(connectedCenter.at<float>(0), connectedCenter.at<float>(1), connectedCenter.at<float>(2));
                }

                KeyFrame* parentKF = kf->GetParent();
                if (parentKF) {
                    cv::Mat parentCenter = parentKF->GetCameraCenter();
                    glVertex3f(cameraCenter.at<float>(0), cameraCenter.at<float>(1), cameraCenter.at<float>(2));
                    glVertex3f(parentCenter.at<float>(0), parentCenter.at<float>(1), parentCenter.at<float>(2));
                }

                auto loopKeyFrames = kf->GetLoopEdges();
                for (auto loopKF : loopKeyFrames) {
                    if (loopKF->mnId < kf->mnId)
                        continue;
                    cv::Mat loopCenter = loopKF->GetCameraCenter();
                    glVertex3f(cameraCenter.at<float>(0), cameraCenter.at<float>(1), cameraCenter.at<float>(2));
                    glVertex3f(loopCenter.at<float>(0), loopCenter.at<float>(1), loopCenter.at<float>(2));
                }
            }
            glEnd();
        }
    }

    void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc) {
        const float width = mCameraSize;
        const float height = width * 0.75;
        const float depth = width * 0.6;

        glPushMatrix();
#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

        glLineWidth(mCameraLineWidth);
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(width, height, depth);
        glVertex3f(0, 0, 0);
        glVertex3f(width, -height, depth);
        glVertex3f(0, 0, 0);
        glVertex3f(-width, -height, depth);
        glVertex3f(0, 0, 0);
        glVertex3f(-width, height, depth);
        glVertex3f(width, height, depth);
        glVertex3f(width, -height, depth);
        glVertex3f(-width, height, depth);
        glVertex3f(-width, -height, depth);
        glVertex3f(-width, height, depth);
        glVertex3f(width, height, depth);
        glVertex3f(-width, -height, depth);
        glVertex3f(width, -height, depth);
        glEnd();

        glPopMatrix();
    }

    void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw) {
        unique_lock<mutex> lock(mMutexCamera);
        mCameraPose = Tcw.clone();
    }

    void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M) {
        if (!mCameraPose.empty()) {
            cv::Mat Rwc(3,3,CV_32F);
            cv::Mat twc(3,1,CV_32F);
            {
                unique_lock<mutex> lock(mMutexCamera);
                Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
                twc = -Rwc * mCameraPose.rowRange(0,3).col(3);
            }

            M.m[0] = Rwc.at<float>(0,0);
            M.m[1] = Rwc.at<float>(1,0);
            M.m[2] = Rwc.at<float>(2,0);
            M.m[3] = 0.0;

            M.m[4] = Rwc.at<float>(0,1);
            M.m[5] = Rwc.at<float>(1,1);
            M.m[6] = Rwc.at<float>(2,1);
            M.m[7] = 0.0;

            M.m[8] = Rwc.at<float>(0,2);
            M.m[9] = Rwc.at<float>(1,2);
            M.m[10] = Rwc.at<float>(2,2);
            M.m[11] = 0.0;

            M.m[12] = twc.at<float>(0);
            M.m[13] = twc.at<float>(1);
            M.m[14] = twc.at<float>(2);
            M.m[15] = 1.0;
        } else {
            M.SetIdentity();
        }
    }

} // namespace ORB_SLAM2
