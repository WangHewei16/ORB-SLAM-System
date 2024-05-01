#include <opencv2/opencv.hpp>
#include "System.h"
#include <string>
#include <chrono>
using namespace std;

int main(int argc, char **argv) {
    string configPath = "/home/stephen/Downloads/ORB_SLAM2/Examples/Monocular/laptop_config.yaml";
    string vocabularyPath = "/home/stephen/Downloads/ORB_SLAM2/Vocabulary/ORBvoc.txt";
    string videoPath = "/home/stephen/Downloads/ORB_SLAM2/Examples/Monocular/myvideo.mp4";
    ORB_SLAM2::System slam(vocabularyPath, configPath, ORB_SLAM2::System::MONOCULAR, true);
    cv::VideoCapture video(videoPath);
    auto startTime = chrono::system_clock::now();
    while (true) {
        cv::Mat originalFrame;
        video >> originalFrame;
        if (!originalFrame.data)
            break;
        cv::Mat resizedFrame;
        cv::resize(originalFrame, resizedFrame, cv::Size(1280, 720));
        auto currentTime = chrono::system_clock::now();
        auto timeElapsed = chrono::duration_cast<chrono::milliseconds>(currentTime - startTime);
        slam.TrackMonocular(resizedFrame, double(timeElapsed.count()) / 1000.0);
    }
}