#include "vienerDeconv.h"

cv::Mat motion_kernel(int angle, int d, int sz)
{
	cv::Mat kern(1, d, CV_32F, cv::Scalar(1));
	double c = cos(angle*Pi / 180);
	double s = sin(angle*Pi / 180);
	float A[2][3] = {{c, -s, 0}, {s, c, 0}};
	int sz2 = sz / 2;
	A[0][2] = sz2 - A[0][0] * (d - 1)*0.5;
	A[1][2] = sz2 - A[1][0] * (d - 1)*0.5;
	cv::Mat M(2, 3, CV_32F, A);

	cv::Mat tmp(d, d, CV_32F);
	cv::Size si(d, d);
	cv::warpAffine(kern, tmp, M, si, CV_INTER_LINEAR);
	return tmp;
}

int main(int argc, char** argv)
{
	int angle = 135;
	int d = 31;
	int sz = 65;

	cv::namedWindow("PSF", cv::WINDOW_NORMAL);
	cv::Mat psf;

	psf = motion_kernel(angle, d, sz);

	cv::imshow("PSF", psf);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 15;
}