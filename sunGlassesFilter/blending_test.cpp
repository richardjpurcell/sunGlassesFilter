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

void alphaBlend(Mat& foreground, Mat& background, Mat& alpha, Mat& outImage);

int main(int argc, char** argv)
{
    Mat foreground = imread("puppets.png");
    Mat background = imread("ocean.png");
    Mat alpha = imread("puppets_alpha.png");

    foreground.convertTo(foreground, CV_32FC3);
    background.convertTo(background, CV_32FC3);

    alpha.convertTo(alpha, CV_32FC3, 1.0 / 255);

    Mat outImage = Mat::zeros(foreground.size(), foreground.type());

    //if alphaBlend function isn't used...
    //multiply(alpha, foreground, foreground);
   // multiply(Scalar::all(1.0) - alpha, background, background);
    //add(foreground, background, outImage);

    alphaBlend(foreground, background, alpha, outImage);

    imshow("alpha blended image", outImage / 255);
    waitKey(0);

    return 0;
}

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