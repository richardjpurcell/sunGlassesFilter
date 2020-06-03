#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>

using namespace cv;
using namespace std;

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

int main()
{
    //images
    Mat foreground = imread("sunglasses_foreground.jpg");
    Mat background = imread("ocean.png");
    Mat alpha = imread("sunglasses_alpha.jpg");
    //bounding rectangle
    Point point1 = Point(100, 100);
    Point point2 = Point(300, 600);
    //calculate sunglasses resize
    int newWidth = point2.x - point1.x;
    int newHeight = (newWidth * foreground.rows)/foreground.cols;
    //resize foreground and alpha to bounding rectangle
    resize(foreground, foreground, Size(newWidth, newHeight), INTER_CUBIC);
    resize(alpha, alpha, Size(newWidth, newHeight), INTER_CUBIC);
    //for testing, draw rectangle
    drawRect(background, point1, point2, true);
    //convert to 32 bit
    foreground.convertTo(foreground, CV_32FC3);
    background.convertTo(background, CV_32FC3);
    alpha.convertTo(alpha, CV_32FC3, 1.0 / 255);
    Mat outImage = Mat::zeros(foreground.size(), foreground.type());
    //prepare region of interest
    Rect roi = Rect(point1.x, point1.y + (point2.y - point1.y) / 3, newWidth, newHeight);
    Mat background_roi = background(roi);
    //blend the images together
    alphaBlend(foreground, background_roi, alpha, outImage);
    //recombine bg and bg_roi
    outImage.copyTo(background(roi));
    imshow("background", background / 255);
    waitKey(0);

    return 0;
}
