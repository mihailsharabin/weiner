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
	char* filename = "____!!!!____";						//input some file!!!
	int angle = 135;
	int d = 31;
	int sz = 65;

	cv::Mat img;
	cv::Mat blurred;
	cv::Mat defocus_psf;
	cv::Mat psf;

	img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);	//reading and preparing an image for 
	img.convertTo(img, CV_32FC1);							//further processing. grayscale - important! (1 channel)
	cv::divide(img, 255.0, img);

	
	cv::namedWindow("initial", cv::WINDOW_AUTOSIZE);		//windows creation
	cv::namedWindow("blurred", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("PSF", cv::WINDOW_NORMAL);
	cv::namedWindow("defocus_psf", cv::WINDOW_NORMAL);

															//initial creation of blurred image (border blur)
	blurred = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);

	int border_blur_rad = 31;												//border blur radious
	
	blur_edge(img, blurred, border_blur_rad);								//border blur

	defocus_psf = defocus_kernel(d, sz);
	
	psf = motion_kernel(angle, d, sz);						//motion psf

	cv::imshow("initial", img);
	cv::imshow("PSF", psf);									//showing the images
	cv::imshow("blurred", blurred);
	cv::imshow("defocus_psf", defocus_psf);

	cv::waitKey(0);

	img.release();											//memory cleaning
	blurred.release();
	psf.release();
	defocus_psf.release();

	cv::destroyAllWindows();								//exit
	return 0;
}

void blur_edge(cv::Mat img, cv::Mat blurred, int d){
	int height = img.rows;
	int width = img.cols;
	cv::Mat img_pad(height + 2 * d, width + 2 * d, CV_32FC1);
	cv::Mat img_blur(height + 2 * d, width + 2 * d, CV_32FC1);
	cv::copyMakeBorder(img, img_pad, d, d, d, d, cv::BORDER_WRAP);
	cv::GaussianBlur(img_pad, img_blur, cvSize(2 * d + 1, 2 * d + 1), -1);
	img_blur = img_blur(cv::Range(d, height + d), cv::Range(d, width + d));
	cv::Mat blur_mat(height, width, CV_32FC1);
	cv::Mat inv_blur_mat(height, width, CV_32FC1);
	int arr[4];
	float tmp;
	for (int i = 0; i < blurred.rows; i++){
		for (int j = 0; j < blurred.cols; j++){
			arr[0] = i; arr[1] = height - 1 - i; arr[2] = j; arr[3] = width - 1 - j;
			tmp = (float)(Min(arr, 4)) / d;
			tmp = (tmp < 1.0) ? tmp : (float)(1.0);
			blur_mat.at<float>(i, j) = tmp;
			inv_blur_mat.at<float>(i, j) = (float)(1.0) - tmp;
		}
	}
	try{
		cv::multiply(img, blur_mat, blurred);
		cv::multiply(img_blur, inv_blur_mat, blur_mat);
		cv::add(blurred, blur_mat, blurred);
	}
	catch (cv::Exception const& e) {
		std::cerr << "OpenCV exception at blur_edge: " << e.what() << std::endl;
	}
	img_pad.release();
	img_blur.release();
	blur_mat.release();
	inv_blur_mat.release();
}

int Min(int* arr, int len){
	if (len == 1){
		return arr[0];
	}
	else {
		return (arr[0] < Min(arr + 1, len - 1)) ? arr[0] : Min(arr + 1, len - 1);
	}
}

cv::Mat defocus_kernel(int d, int sz){
	cv::Mat kern = cv::Mat::zeros(sz, sz, CV_8UC1);
	cv::circle(kern, cv::Point(sz, sz), d, 255, -1, CV_AA, 1);
	kern.convertTo(kern, CV_32F);
	cv::divide(kern, 255.0, kern);
	return kern;
}