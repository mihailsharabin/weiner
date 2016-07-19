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
	char* filename = "____________";
	int angle = 135;
	int d = 31;
	int sz = 65;

	cv::Mat img;

	//preparing an image for further processing
	img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	img.convertTo(img, CV_32F);
	cv::divide(img, 255.0, img);

	cv::namedWindow("initial", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("blurred", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("PSF", cv::WINDOW_NORMAL);
	cv::Mat psf;


	psf = motion_kernel(angle, d, sz);

	cv::imshow("PSF", psf);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 15;
}

void blur_edge(cv::Mat img, cv::Mat blurred, int d){
	if (!d)
		d = 31;
	int height = img.rows;
	int width = img.cols;
	cv::Mat img_pad(height + 2 * d, width + 2 * d, CV_32F);
	cv::Mat img_blur(height + 2 * d, width + 2 * d, CV_32F);
	cv::copyMakeBorder(img, img_pad, d, d, d, d, cv::BORDER_WRAP);
	cv::GaussianBlur(img_pad, img_blur, cvSize(2 * d + 1, 2 * d + 1), -1);
	img_blur = img_blur(cv::Range(d, height + d), cv::Range(d, width + d));
	cv::Mat prefinal(height, width, CV_32F);
	cv::Mat anti_pref(height, width, CV_32F);
	int arr[4];
	float tmp;
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			arr[0] = i; arr[1] = height - 1 - i; arr[2] = j; arr[3] = width - 1 - j;
			tmp = (float)(Min(arr, 4)) / d;
			tmp = (tmp < 1.0) ? tmp : (float)(1.0);
			prefinal.at<float>(i, j) = tmp;
			anti_pref.at<float>(i, j) = (float)(1.0) - tmp;
			blurred.at<float>(i, j) = img.at<float>(i, j) * tmp + img_blur.at<float>(i, width - j - 1) * ((float)(1.0) - tmp);
		}
	}
	//	cv::multiply(img, prefinal, blurred);
	//	cv::multiply(img_blur, anti_pref, prefinal);
	//	cv::add(blurred, prefinal, blurred);
	img_pad.release();
	img_blur.release();
	prefinal.release();
	anti_pref.release();
}

int Min(int* arr, int len){
	if (len == 1){
		return arr[0];
	}
	else {
		return (arr[0] < Min(arr + 1, len - 1)) ? arr[0] : Min(arr + 1, len - 1);
	}
}
