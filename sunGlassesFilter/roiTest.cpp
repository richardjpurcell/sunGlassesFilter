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
    int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();
    float* fptr = reinterpret_cast<float*>(foreground.data);
    float* bptr = reinterpret_cast<float*>(background.data);
    float* aptr = reinterpret_cast<float*>(alpha.data);
    float* outPtr = reinterpret_cast<float*>(outImage.data);

    for (int i = 0; i < numberOfPixels; i++, outPtr++, fptr++, bptr++, aptr++)
    {
        *outPtr = (*fptr) * (*aptr) + (*bptr) * (1 - *aptr);
    }
}

int main()
{
    Mat foreground = imread("sunglasses.jpg");
    Mat background = imread("ocean.png");
    Mat alpha = imread("sunglasses_alpha.jpg");

    int height = foreground.rows;
    int width = foreground.cols;
    resize(foreground, foreground, Size(width / 2, height / 2), INTER_CUBIC);
    resize(alpha, alpha, Size(width / 2, height / 2), INTER_CUBIC);

    foreground.convertTo(foreground, CV_32FC3);
    background.convertTo(background, CV_32FC3);
    alpha.convertTo(alpha, CV_32FC3, 1.0 / 255);

    Mat outImage = Mat::zeros(foreground.size(), foreground.type());

    cout << foreground.size() << endl;
    cout << background.size() << endl;

    Rect roi = Rect(0, 0, foreground.cols, foreground.rows);
    Mat background_roi = background(roi);

    alphaBlend(foreground, background_roi, alpha, outImage);

    imshow("alpha blended", outImage / 255);
    waitKey(0);

    return 0;
}
