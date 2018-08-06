//
// Created by Michael Zhang on 2018/8/3.
//

#ifndef EYEOC_SVM_UTIL_H
#define EYEOC_SVM_UTIL_H

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

void ReadImages(Mat &dst, const string dir, const unsigned int label);

#endif //EYEOC_SVM_UTIL_H
