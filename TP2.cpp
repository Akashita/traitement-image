#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat filtreM(Mat input) {
    Mat M = (Mat_<float>(3, 3) <<
            1.0 / 16, 2.0 / 16, 1.0 / 16,
            2.0 / 16, 4.0 / 16, 2.0 / 16,
            1.0 / 16, 2.0 / 16, 1.0 / 16);

    Mat output;
    filter2D(input, output, CV_32FC1, M);
    return output;
}

Mat filtreMedian(Mat input) {
    Mat output;
    medianBlur(input, output, 3);
    return output;
}

Mat filtreLaplacien1(Mat input, float alpha) {
    Mat M = (Mat_<float>(3, 3) <<
            0, 1, 0,
            1, -4, 1,
            0, 1, 0);

    M = alpha * M;

    Mat ID = (Mat_<float>(3, 3) <<
            0, 0, 0,
            0, 1, 0,
            0, 0, 0);

    M = ID - M;

    Mat output;
    filter2D(input, output, CV_32FC1, M);
    return output;
}

Mat filtreSobelX(Mat input) {
    Mat Sobel = (Mat_<float>(3, 3) <<
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1);

    Mat output;
    filter2D(input, output, CV_32FC1, Sobel, Point(0, 0), 0);
    return output;
}

Mat filtreSobelY(Mat input) {
    Mat Sobel = (Mat_<float>(3, 3) <<
            -1, -2, -1,
            0, 0, 0,
            1, 2, 1);

    Mat output;
    filter2D(input, output, CV_32FC1, Sobel, Point(0, 0), 0);
    return output;
}

Mat filtreGradient(Mat input) {
    Mat gradX = filtreSobelX(input);
    Mat gradY = filtreSobelY(input);

    Mat gradient = Mat(input.rows, input.cols, CV_32FC1);
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            gradient.at<float>(i, j) = sqrt(pow(gradX.at<float>(i, j), 2.0) + pow(gradY.at<float>(i, j), 2.0));
        }
    }
    return gradient;
}

Mat rehaussementContraste(Mat input, int alpha) {
    Mat output;

    Mat matrice = (Mat_<float>(3, 3) << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    Mat laplacien = (Mat_<float>(3, 3) << 0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0);
    Mat result = matrice - alpha * laplacien;

    filter2D(input, output, -1, result);
    return output;
}

Mat detectionContoursMarrHildreth(Mat input, int seuil) {
    Mat gradX = filtreLaplacien1(input, 1.0);
    Mat gradY = filtreLaplacien1(input, 1.0);

    Mat gradient = Mat(input.rows, input.cols, CV_32FC1);
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            gradient.at<float>(i, j) = sqrt(pow(gradX.at<float>(i, j), 2.0) + pow(gradY.at<float>(i, j), 2.0));
        }
    }

    Mat seuillage;
    threshold(gradient, seuillage, seuil, 255, THRESH_BINARY);

    Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
    Mat output;
    dilate(seuillage, output, element);

    return output;
}

int main(int argc, char *argv[]) {
    namedWindow("Filter");
    Mat input = imread(argv[1]);
    if (input.channels() == 3)
        cv::cvtColor(input, input, COLOR_BGR2GRAY);

    input.convertTo(input, CV_32FC1);

    int alpha = 20;
    createTrackbar("alpha (en %)", "Filter", nullptr, 100, NULL);
    setTrackbarPos("alpha (en %)", "Filter", alpha);
    while (true) {
        input.convertTo(input, CV_32FC1);

        alpha = getTrackbarPos("alpha (en %)", "Filter");
        int keycode = waitKey(50);
        //cout << keycode << endl;
        switch (keycode) {
            case 97:
                input = filtreM(input);
                break;
            case 109:
                input = filtreMedian(input);
                break;
            case 115:
                input = filtreLaplacien1(input, alpha / 100.0);
                break;
            case 120:
                input = filtreSobelX(input);
                break;
            case 121:
                input = filtreSobelY(input);
                break;
            case 103:
                input = filtreGradient(input);
                break;
            case 108:
                input = detectionContoursMarrHildreth(input, alpha);
                break;
        }

        int asciicode = keycode & 0xff;
        if (asciicode == 'q') break;

        input.convertTo(input, CV_8U);
        imshow("Filter", input);            // l'affiche dans la fenêtre
    }
    imwrite("result.png", input);          // sauvegarde le résultat
}
