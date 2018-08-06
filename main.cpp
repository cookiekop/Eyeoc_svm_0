#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "util.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

string openRImage = "dataset/openRightEyes/";
string openLImage = "dataset/openLeftEyes/";
string closeRImage = "dataset/closedRightEyes/";
string closeLImage = "dataset/closedLeftEyes/";

double cost_time_;
clock_t start_time_;
clock_t end_time_;

int main()
{
    //--------------------- 1. Set up training data ---------------------------------------
    Mat data_or, data_ol, data_cr, data_cl;
    ReadImages(data_or, openRImage, 1);
    ReadImages(data_ol, openLImage, 1);
    ReadImages(data_cr, closeRImage, 0);
    ReadImages(data_cl, closeLImage, 0);

    //cout << data_or.at<unsigned int>(0, 24*24) << endl;
    int total_num = data_cl.rows + data_cr.rows + data_ol.rows + data_or.rows;
    Mat temp_data, open_data;
    vconcat(data_or, data_ol, open_data);
    vconcat(open_data, data_cr, temp_data);
    vconcat(temp_data, data_cl, temp_data);
    //vconcat(data_or, data_cr, temp_data);

    /*int open_num = open_data.rows;
    Mat openData(Size(24 * 24, open_num), CV_32FC1), openLabel;
    openLabel = cv::Mat::ones(Size(1, open_num), CV_32SC1);
    for (int i=0; i<open_num; ++i) {
        for (int x=0; x<24; ++x)
            for (int y=0; y<24; ++y)
                openData.at<float>(i, x*24+y) = open_data.at<float>(i, x*24+y);
    }*/

    vector<int> index;
    Mat labels(Size(1, total_num), CV_32SC1), Data(Size(24 * 24, total_num), CV_32FC1);
    for (int i=0; i<total_num; ++i)
        index.push_back(i);
    randShuffle(index);

    for (int i=0; i<total_num; ++i) {
        labels.at<unsigned int>(index[i], 0) = temp_data.at<unsigned int>(i, 24*24);
        for (int x=0; x<24; ++x)
            for (int y=0; y<24; ++y)
                Data.at<float>(index[i], x*24+y) = temp_data.at<float>(i, x*24+y);
    }

    int train_num = total_num *2 /3;
    int test_num = total_num - train_num;
    Mat train_label, trainData;
    train_label = labels.rowRange(0, train_num);
    trainData = Data.rowRange(0, train_num);

    cout << trainData.rows << " " << trainData.cols << endl;
    cout << train_label.rows << " " << train_label.cols << endl;

    //------------------------ 2. Set up the support vector machines parameters --------------------
    /*Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::ONE_CLASS);
    svm->setKernel(SVM::RBF);
    //svm->setDegree(10.0);
    svm->setGamma(0.01);
    //svm->setCoef0(1.0);
    //svm->setC(2.0);
    svm->setNu(0.5);
    //svm->setP(0.1);
    svm->setTermCriteria(TermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON));*/


    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::NU_SVC);
    svm->setKernel(SVM::RBF);
    //svm->setDegree(10.0);
    svm->setGamma(0.025);
    //svm->setCoef0(1.0);
    //svm->setC(2.0);
    svm->setNu(0.15);
    //svm->setP(0.1);
    svm->setTermCriteria(TermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON));

    //------------------------ 3. Train the svm ----------------------------------------------------
    /*cout << "Starting training process" << endl;
    start_time_ = clock();
    svm->train(openData, ROW_SAMPLE, openLabel);
    end_time_ = clock();
    cost_time_ = (end_time_ - start_time_) / CLOCKS_PER_SEC;
    cout << "Finished training process...cost " << cost_time_ << " seconds..." << endl;*/
    cout << "Starting training process" << endl;
    start_time_ = clock();
    svm->train(trainData, ROW_SAMPLE, train_label);
    end_time_ = clock();
    cost_time_ = (end_time_ - start_time_) / CLOCKS_PER_SEC;
    cout << "Finished training process...cost " << cost_time_ << " seconds..." << endl;

    //------------------------ 4. save the svm ----------------------------------------------------
    svm->save("Eyeoc_svm.xml");
    cout << "saved" << endl;


    //------------------------ 5. load the svm ----------------------------------------------------
    Ptr<SVM> svm1 = StatModel::load<SVM>("Eyeoc_svm.xml");
    Mat test_data = imread("9.jpg",  IMREAD_GRAYSCALE);
    test_data.convertTo(test_data, CV_32F, 1.0/255.0);
    //resize(test_data, test_data, Size(24,24));
    //cout << test_data << endl;
    Mat test(Size(24*24,1), CV_32F);
    for (int i=0; i<24; ++i)
        for(int j=0; j<24; ++j)
            test.at<float>(i * 24 + j) = test_data.at<float>(i, j);
    cout << svm1->predict(test) << endl;


    //------------------------ 6. read the test dataset -------------------------------------------
    float t_count = 0;
    for (int i = 0; i < trainData.rows; i++) {
        Mat sample = trainData.row(i);
        float res = svm1->predict(sample);
        res = std::abs(res - train_label.at<unsigned int>(i, 0)) <= FLT_EPSILON ? 1.f : 0.f;
        t_count += res;
    }
    //cout << "train correct count = " << t_count << endl;
    cout << "train error rate " << (train_num - t_count + 0.0) / train_num * 100.0 << "%....\n";

    Mat testData;
    Mat tLabel;
    testData = Data.rowRange(train_num, total_num);
    tLabel = labels.rowRange(train_num, total_num);


    float count = 0;
    for (int i = 0; i < testData.rows; i++) {
        Mat sample = testData.row(i);
        float res = svm1->predict(sample);
        res = std::abs(res - tLabel.at<unsigned int>(i, 0)) <= FLT_EPSILON ? 1.f : 0.f;
        count += res;
    }
    //cout << "correct count = " << count << endl;
    cout << "error rate " << (test_num - count + 0.0) / test_num * 100.0 << "%....\n";
    return 0;
}