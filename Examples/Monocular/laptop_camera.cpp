#include <opencv2/opencv.hpp>
#include "System.h"
#include <chrono>
using namespace std;

int main(int argc, char **argv) {
    ORB_SLAM2::System slamSystem("/home/stephen/Downloads/ORB_SLAM2/Vocabulary/ORBvoc.txt", "/home/stephen/Downloads/ORB_SLAM2/Examples/Monocular/laptop_config.yaml", ORB_SLAM2::System::MONOCULAR, true);
    cv::VideoCapture camera(0);
    camera.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    auto startTime = chrono::system_clock::now();
    while (true) {
        cv::Mat img;
        camera >> img;
        auto currentTime = chrono::system_clock::now();
        auto elapsedTime = chrono::duration_cast<chrono::milliseconds>(currentTime - startTime);
        slamSystem.TrackMonocular(img, double(elapsedTime.count())/1000.0);
    }
}
