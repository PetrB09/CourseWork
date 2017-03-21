#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <fstream>
#include <vector_types.h>

#include <cv.h>
#include <HandlingImgOnGPU.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector_types.h>
#include <opencv2/dnn.hpp>


using namespace std;
using namespace cv;
using namespace cuda;
//using namespace cv::dnn

int main(int argc, char **argv)
{
	string inputPathL = "inputL.png";
	string inputPathR = "inputR.png";
	string outputPathL = "outputL.png";
	string outputPathR = "outputR.png";
	char* logPath = (char*)"tmp.log";
	ofstream logOut;
	logOut.open(logPath);

	logOut<< "reading files " << inputPathL << " " << inputPathR;

	cv::Mat LeftPic = cv::imread(inputPathL, CV_LOAD_IMAGE_COLOR);//reading image from folder
	cv::Mat RightPic = cv::imread(inputPathR, CV_LOAD_IMAGE_COLOR);
	if(LeftPic.empty() || RightPic.empty())
		logOut << "Files not readed\n";
	else
		logOut << "Files succesfuly readed\n";
	uint Stime = clock();
	cv::cuda::GpuMat GPULeft(LeftPic); //loading inage to GPU
	cv::cuda::GpuMat GPURight(RightPic);

	logOut << "size of image " << GPULeft.size() << "\n";
	logOut << "type of element " << GPULeft.type() << "\n";
	logOut << "size of element " << GPULeft.elemSize() << "\n";
	logOut << "size element on channel " << GPULeft.elemSize1() << "\n";

	cuda::bilateralFilter(GPULeft, GPULeft, 5, 150, 150); //Handling image
	cuda::bilateralFilter(GPURight, GPURight, 5, 150, 150);

	GPULeft.download(LeftPic); //return image to CPU
	GPURight.download(RightPic);

	uint Ftime = clock();

	bool LeftWrited = imwrite(outputPathL, LeftPic, vector<int>(CV_IMWRITE_PNG_COMPRESSION, 0)); //save image
	bool RightWrited = imwrite(outputPathR, RightPic, vector<int>(CV_IMWRITE_PNG_COMPRESSION, 0));

	vector<uchar4***> pointers;


	if(!LeftWrited && !RightWrited)
		logOut << "Error of writing\n";
	else
		logOut << "Images succesfuly writed\n";
	logOut << "runtime: "<< (float)(Ftime - Stime)/CLOCKS_PER_SEC;
	logOut.close();
	return 0;
}
