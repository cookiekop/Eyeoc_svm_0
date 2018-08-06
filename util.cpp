//
// Created by Michael Zhang on 2018/8/3.
//
#include "util.h"

const double zoom=1.37;

void rotate(const Mat &src, Mat &rot, double angle) {
    Point2f center(src.cols / 2, src.rows / 2);
    double cos, sin;
    int nW, nH;
    Mat rot_mat = getRotationMatrix2D(center, angle, 1.0);
    cos = abs(rot_mat.at<double>(0, 0));
    sin = abs(rot_mat.at<double>(0, 1));
    nW = int((src.rows * sin) + (src.cols * cos));
    nH = int((src.rows * cos) + (src.cols * sin));
    rot_mat.at<double>(0, 2) += nW / 2.0 - center.x;
    rot_mat.at<double>(1, 2) += nH / 2.0 - center.y;
    warpAffine(src, rot, rot_mat, Size(nW,nH));
}

void rotateImage(const Mat &src, vector<Mat> &out) {
    Mat src_zoom;
    resize(src, src_zoom, Size(src.cols * zoom, src.rows * zoom));
    for (double angle = -30; angle<=30; angle = angle +5) {
        Mat rot;
        rotate(src_zoom, rot, angle);
        out.push_back(rot.clone());
    }
}

void ReadImages(Mat &dst, const string dir, const unsigned int label) {
    string pattern = dir + "*.jpg";
    vector<String> files;
    glob(pattern ,files);
    dst.create(Size(24*24 + 1, files.size() * 13), CV_32FC1);
    for (int i=0; i<files.size(); ++i) {
        Mat img = imread(files[i], IMREAD_GRAYSCALE);
        vector<Mat> img_rot;
        rotateImage(img, img_rot);
        for (int j=0; j<img_rot.size(); ++j) {
            Point2f org(img_rot[j].cols /2 - 12, img_rot[j].rows /2 -12);
            Rect roi(org, Size(24,24));
            Mat img_roi = img_rot[j](roi);
            /*imshow("t", img_roi);
            waitKey(0);*/
            img_roi.convertTo(img_roi, CV_32F, 1.0/255.0);
            for (int x=0; x<24; ++x)
                for (int y=0; y<24; ++y)
                    dst.at<float>(i * 13 + j, x*24+y) = img_roi.at<float>(x,y);
            dst.at<unsigned int>(i * 13 + j, 24*24) = label;
        }
    }
}
