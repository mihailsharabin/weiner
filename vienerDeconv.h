#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cmath>

#define Pi 3.14159265
cv::Mat motion_kernel(int angle, int d, int sz);
void blur_edge(cv::Mat img, cv::Mat blurred, int d);
int Min(int* arr, int len);