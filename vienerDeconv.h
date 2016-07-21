
#define _USE_MATH_DEFINES

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cmath>

cv::Mat motion_kernel(int angle, int d, int sz);
void blur_edge(cv::Mat img, cv::Mat blurred, int d);
int Min(int* arr, int len);
cv::Mat defocus_kernel(int d, int sz);
cv::Mat deconvolve(cv::Mat img, bool defocus, int d, int ang, int noise, int sz);
void roll_mat(cv::Mat img, int x, int y);
