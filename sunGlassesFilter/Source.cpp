#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace std;

int main(int argc, const char** argv)
{
    Mat frame;

    VideoCapture cap;

    int deviceID = 0;
    int apiID = cv::CAP_ANY;

    cap.open(deviceID, apiID);

    if (!cap.isOpened())
    {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

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

        imshow("Live", frame);
        if (waitKey(5) >= 0)
            break;
    }


    return 0;
}

