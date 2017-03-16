#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <cv.h>
//#include <HandlingImgOnGPU.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;
using namespace cuda;

int main(int argc, char **argv)
{
	string inputPathL = "inputL.png";
	string inputPathR = "inputR.png";
	string outputPathL = "outputL.png";
	string outputPathR = "outputR.png";

	cv::Mat LeftPic = imread(inputPathL, -1);//reading image from folder
	cv::Mat RightPic = imread(inputPathR, -1);

	cv::cuda::GpuMat GPULeft(LeftPic); //loading inage to GPU
	cv::cuda::GpuMat GPURight(RightPic);

	cuda::bilateralFilter(GPULeft, GPULeft, 5, 150, 150); //Handling image
	cuda::bilateralFilter(GPURight, GPURight, 5, 150, 150);

	GPULeft.download(LeftPic); //return image to CPU
	GPURight.download(RightPic);

	imwrite(outputPathL, LeftPic, vector<int>(-1)); //save image
	imwrite(outputPathR, RightPic, vector<int>(-1));

	return 0;
}
