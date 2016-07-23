#include "vienerDeconv.h"

int main(int argc, char** argv)
{
	char* filename = "____!!!!____";						//input some file!!!
	int angle = 455;
	int d = 22;
	int sz = 65;
	int noise = 25;

	cv::Mat img;
	cv::Mat defocus_psf;
	cv::Mat psf;
	cv::Mat deconvolved;

	img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);	//reading and preparing an image for 
	img.convertTo(img, CV_32FC1);							//further processing. grayscale - important! (1 channel)
	cv::divide(img, 255.0, img);

	
	cv::namedWindow("initial", cv::WINDOW_AUTOSIZE);		//windows creation
	cv::namedWindow("psf", cv::WINDOW_NORMAL);
	cv::namedWindow("defocus_psf", cv::WINDOW_NORMAL);
	cv::namedWindow("deconvolved", cv::WINDOW_AUTOSIZE);
															//initial creation of blurred image (border blur)
	blurred = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
	
	deconvolved = deconvolve(img, false, d, angle, noise, sz);				//final function (a.k.a. update)

	defocus_psf = defocus_kernel(d, sz);									//defocus psf
	
	psf = motion_kernel(angle, d, sz);						//motion psf

	cv::imshow("initial", img);
	cv::imshow("psf", psf);									//showing the images
	cv::imshow("defocus_psf", defocus_psf);
	cv::imshow("deconvolved", deconvolved);

	cv::waitKey(0);

	img.release();											//memory cleaning
	psf.release();
	defocus_psf.release();
	deconvolved.release();

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

cv::Mat motion_kernel(int angle, int d, int sz){
	cv::Mat kern = cv::Mat::ones(1, d, CV_32FC1);
	cv::Mat tmp = cv::Mat::zeros(d, d, CV_32FC1);
	cv::Mat move_mat = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, d / 2);
	cv::warpAffine(kern, tmp, move_mat, tmp.size());
	cv::Point center(d / 2, d / 2);
	cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
	cv::warpAffine(tmp, tmp, M, tmp.size());
	kern.release();
	move_mat.release();
	M.release();
	return tmp;
}

//deconvolve works with images, read in grayscale mode, 
//converted to CV_32F and divided by 255
cv::Mat deconvolve(cv::Mat img, bool defocus, int d, int ang, int noise, int sz){
	double snr = pow(10, -0.1*noise);
	blur_edge(img, img, 31);
	cv::Mat IMG;
	cv::dft(img, IMG, cv::DFT_COMPLEX_OUTPUT);
	cv::Mat psf;

	if (defocus)
		psf = defocus_kernel(d, sz);
	else
		psf = motion_kernel(ang, d, sz);

	cv::namedWindow("psf", cv::WINDOW_NORMAL);
	cv::imshow("psf", psf);

	cv::divide(psf, cv::sum(psf)[0], psf);
	cv::Mat psf_pad = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
	int kh = psf.rows;
	int kw = psf.cols;
	cv::Mat roi(psf_pad(cv::Rect(0, 0, kw, kh)));
	psf.copyTo(roi);
	cv::Mat PSF;
	cv::dft(psf_pad, PSF, cv::DFT_COMPLEX_OUTPUT, kh);
	cv::Mat PSF2 = cv::Mat::zeros(PSF.rows, PSF.cols, PSF.type());					//formula solution
	cv::mulSpectrums(PSF, PSF, PSF2, 0, true);
	cv::Mat mat_arr[2];
	cv::split(PSF2, mat_arr);
	mat_arr[0] += snr;
	mat_arr[1] = mat_arr[0];
	cv::merge(mat_arr, 2, PSF2);
	cv::divide(IMG, PSF2, IMG);
	cv::mulSpectrums(IMG, PSF, IMG, 0, true);
	cv::Mat result(img.rows, img.cols, CV_32FC1);
	cv::idft(IMG, result, cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);
	roll_mat(result, kh, kw);
	IMG.release();
	psf.release();
	psf_pad.release();
	PSF.release();
	PSF2.release();
	roi.release();
	mat_arr[0].release();
	mat_arr[1].release();
	return result;
}

void roll_mat(cv::Mat img, int x, int y){
	cv::Mat tmp = cv::Mat(img.rows, img.cols, img.type());	
	cv::Mat tmp1;
	tmp1 = img(cv::Rect(0, 0, img.cols - y, img.rows - x));
	tmp1.copyTo(tmp(cv::Rect(y, x, img.cols - y, img.rows - x)));
	tmp1 = img(cv::Rect(img.cols - y, 0, y, img.rows - x));
	tmp1.copyTo(tmp(cv::Rect(0, x, y, img.rows - x)));
	tmp1 = img(cv::Rect(0, img.rows - x, img.cols - y, x));
	tmp1.copyTo(tmp(cv::Rect(y, 0, img.cols - y, x)));
	tmp1 = img(cv::Rect(img.cols - y, img.rows - x, y, x));
	tmp1.copyTo(tmp(cv::Rect(0, 0, y, x)));
	tmp1.release(); 
	tmp.copyTo(img);
	tmp.release();
}
