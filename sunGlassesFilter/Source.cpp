#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;

#define BLUR 15 //must be odd number

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.5;
const cv::Scalar meanVal(104.0, 177.0, 123.0);
Mat sunglassesRGB = imread("sunglasses_foreground.jpg");
Mat sunglassesA = imread("sunglasses_alpha.jpg");

#define CAFFE

const std::string caffeConfigFile = "./data/models/deploy.prototxt";
const std::string caffeWeightFile = "./data/models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

const std::string tensorflowConfigFile = "./data/models/opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = "./data/models/opencv_face_detector_uint8.pb";

void detectFaceOpenCVDNN(Net net, Mat& frameOpenCVDNN);

void alphaBlend(Mat& foreground, Mat& background, Mat& alpha, Mat& outImage)
{
    //get rid of roi information in header
    Mat temp = background.clone();
    //create pointers to Mat arrays
    int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();
    float* fptr = reinterpret_cast<float*>(foreground.data);
    float* bptr = reinterpret_cast<float*>(temp.data);
    float* aptr = reinterpret_cast<float*>(alpha.data);
    float* outPtr = reinterpret_cast<float*>(outImage.data);

    for (int i = 0; i < numberOfPixels; i++, outPtr++, fptr++, bptr++, aptr++)
    {
        *outPtr = (*fptr) * (*aptr) + (*bptr) * (1 - *aptr);
    }
}

void drawRect(Mat& img, Point& point1, Point& point2, bool showRect)
{
    Mat temp = img.clone();
    rectangle(img, point1, point2, Scalar(255, 255, 0), 4);
}

void sunglassesProcess(Mat& foreground, Mat& alpha, Point* points)
{
    foreground = sunglassesRGB.clone();
    alpha = sunglassesA.clone();

    //calculate sunglasses resize
    int newWidth = points[1].x - points[0].x;
    int newHeight = (newWidth * foreground.rows) / foreground.cols;
    //preblur foreground and alpha to help resize look less jaggy
    GaussianBlur(foreground, foreground, Size(BLUR, BLUR), 0, 0);
    GaussianBlur(alpha, alpha, Size(BLUR, BLUR), 0, 0);
    //resize foreground and alpha to bounding rectangle

    if ((points[1].x - points[0].x) > 0)
    {
        resize(foreground, foreground, Size(newWidth, newHeight), INTER_AREA);
        resize(alpha, alpha, Size(newWidth, newHeight), INTER_CUBIC);
    }
}

void blendImages(Mat& foreground, Mat& background, Mat& alpha, Mat& out, Point* points)
{
    //calculate sunglasses resize
    int newWidth = points[1].x - points[0].x;
    int newHeight = (newWidth * foreground.rows) / foreground.cols;
    //convert to 32 bit
    foreground.convertTo(foreground, CV_32FC3);
    background.convertTo(background, CV_32FC3);
    alpha.convertTo(alpha, CV_32FC3, 1.0 / 255);

    if ((points[1].x - points[0].x) > 0)
    {
        out = Mat::zeros(foreground.size(), foreground.type());
        //prepare region of interest
        Rect roi = Rect(points[0].x, points[0].y + 2 * (points[1].y - points[0].y) / 7, newWidth, newHeight);
        Mat background_roi = background(roi);
        //blend the images together
        alphaBlend(foreground, background_roi, alpha, out);
        //recombine bg and bg_roi
        out.copyTo(background(roi));
        //imshow("Sunglass Cam", background / 255);
    }
}

void detectFaceOpenCVDNN(Net net, Mat& frameOpenCVDNN, Point *points)
{
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;
#ifdef CAFFE
    cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
#else
    cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);
#endif

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > confidenceThreshold)
        {
            points[0].x = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            points[0].y = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            points[1].x = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            points[1].y = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
        }
    }

}

int main(int argc, const char** argv)
{
#ifdef CAFFE
    Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
#else
    Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
#endif

    //images
    Mat foreground = sunglassesRGB.clone();
    Mat alpha = sunglassesA.clone();
    Mat frame;
    Mat out;
    //camera setup
    VideoCapture cap;
    int deviceID = 0;
    int apiID = cv::CAP_ANY;
    cap.open(deviceID, apiID);
    if (!cap.isOpened())
    {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    //DNN setup
    double tt_opencvDNN = 0;
    double fpsOpencvDNN = 0;
    //corner points of face rectangle
    Point points[2];

    cout << "start grabbing frames" << endl
        << "Press any key to terminate" << endl;
    for (;;)
    {
        cap.read(frame);

        if (frame.empty())
        {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        double t = cv::getTickCount();
        detectFaceOpenCVDNN(net, frame, points);
        //face rectangle for testing
        //cv::rectangle(frame, points[0], points[1], cv::Scalar(0, 255, 0), 2, 4);
        sunglassesProcess(foreground, alpha, points);
        blendImages(foreground, frame, alpha, out, points);
        tt_opencvDNN = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        fpsOpencvDNN = 1 / tt_opencvDNN;
        
        imshow("Sunglass Cam", frame / 255);
        if (waitKey(5) >= 0)
            break;
    }

    return 0;
}



