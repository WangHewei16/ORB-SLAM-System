#include "Converter.h"
#include <unistd.h>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2 {

    std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat &Descriptors) {
        std::vector<cv::Mat> descVector;
        descVector.reserve(Descriptors.rows);
        int idx = 0;
        while (idx < Descriptors.rows) {
            descVector.push_back(Descriptors.row(idx));
            ++idx;
        }
        return descVector;
    }

    g2o::SE3Quat Converter::toSE3Quat(const cv::Mat &cvTransform) {
        Eigen::Matrix<double, 3, 3> rotMat;
        rotMat << cvTransform.at<float>(0, 0), cvTransform.at<float>(0, 1), cvTransform.at<float>(0, 2),
                cvTransform.at<float>(1, 0), cvTransform.at<float>(1, 1), cvTransform.at<float>(1, 2),
                cvTransform.at<float>(2, 0), cvTransform.at<float>(2, 1), cvTransform.at<float>(2, 2);

        Eigen::Matrix<double, 3, 1> transVec(cvTransform.at<float>(0, 3), cvTransform.at<float>(1, 3), cvTransform.at<float>(2, 3));
        return g2o::SE3Quat(rotMat, transVec);
    }

    cv::Mat Converter::toCvMat(const g2o::SE3Quat &se3) {
        Eigen::Matrix<double, 4, 4> matrixEigen = se3.to_homogeneous_matrix();
        return toCvMat(matrixEigen);
    }

    cv::Mat Converter::toCvMat(const g2o::Sim3 &sim3) {
        Eigen::Matrix3d rot = sim3.rotation().toRotationMatrix();
        Eigen::Vector3d trans = sim3.translation();
        double scale = sim3.scale();
        return toCvSE3(scale * rot, trans);
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 4, 4> &mat) {
        cv::Mat matCV(4, 4, CV_32F);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                matCV.at<float>(i, j) = mat(i, j);
            }
        }
        return matCV.clone();
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix3d &mat) {
        cv::Mat matCV(3, 3, CV_32F);
        int i = 0, j = 0;
        while (i < 3) {
            j = 0;
            while (j < 3) {
                matCV.at<float>(i, j) = mat(i, j);
                ++j;
            }
            ++i;
        }
        return matCV.clone();
    }

    cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 3, 1> &vec) {
        cv::Mat matCV(3, 1, CV_32F);
        int i = 0;
        while (i < 3) {
            matCV.at<float>(i) = vec(i);
            ++i;
        }
        return matCV.clone();
    }

    cv::Mat Converter::toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t) {
        cv::Mat matCV = cv::Mat::eye(4, 4, CV_32F);
        int i = 0, j;
        while (i < 3) {
            j = 0;
            while (j < 3) {
                matCV.at<float>(i, j) = R(i, j);
                ++j;
            }
            matCV.at<float>(i, 3) = t(i);
            ++i;
        }
        return matCV.clone();
    }

    Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Mat &cvVec) {
        return Eigen::Matrix<double, 3, 1>(cvVec.at<float>(0), cvVec.at<float>(1), cvVec.at<float>(2));
    }

    Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Point3f &point) {
        return Eigen::Matrix<double, 3, 1>(point.x, point.y, point.z);
    }

    Eigen::Matrix<double, 3, 3> Converter::toMatrix3d(const cv::Mat &cvMat) {
        Eigen::Matrix<double, 3, 3> mat;
        mat << cvMat.at<float>(0, 0), cvMat.at<float>(0, 1), cvMat.at<float>(0, 2),
                cvMat.at<float>(1, 0), cvMat.at<float>(1, 1), cvMat.at<float>(1, 2),
                cvMat.at<float>(2, 0), cvMat.at<float>(2, 1), cvMat.at<float>(2, 2);
        return mat;
    }

    std::vector<float> Converter::toQuaternion(const cv::Mat &M) {
        Eigen::Matrix<double, 3, 3> eigenMatrix = toMatrix3d(M);
        Eigen::Quaterniond quat(eigenMatrix);
        return std::vector<float>{quat.x(), quat.y(), quat.z(), quat.w()};
    }

}
