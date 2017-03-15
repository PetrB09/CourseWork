#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <cv.h>
#include <HandlingImgOnGPU.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <cuda.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	string inputPathL = "inputL.png";
	string inputPathR = "inputR.png";

	cv::Mat LeftPic = imread(inputPathL, -1);
	cv::Mat RightPic = imread(inputPathR, -1);


	cv::cuda::GpuMat GPULeft;
	cv::cuda::GpuMat GPURight;

	return 0;
}
