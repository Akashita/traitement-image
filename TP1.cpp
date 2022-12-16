#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

vector<double> histogramme(Mat image)
{
    vector<double> hist(256);
    double max_value = 0;

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            uchar pixel = image.at<uchar>(i, j);
            hist[pixel]++;
            max_value = max(max_value, hist[pixel]);
        }
    }

    for (int c = 0; c < hist.size(); c++) {
        hist[c] /= max_value;
    }

    return hist;
}

vector<double> histogramme_cumule( const vector<double>& h_I )
{
    vector<double> histCumul;
    double histSum = 0;
    for (int i = 0; i < h_I.size(); i++){
        histSum += h_I[i];
    }
    double lastEntry = 0;
    for (int i = 0; i < h_I.size(); i++){
        histCumul.push_back((lastEntry + h_I[i]) / histSum);
        lastEntry += h_I[i];
    }
    return histCumul;
}

void egalisation(Mat image)
{
    vector<double> hist = histogramme_cumule(histogramme(image));
    for(int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            int color = (int) image.at<uchar>(row, col);
            image.at<uchar>(row, col) = (int) (255 * hist[color]);
        }
    }
}

void tramage_floyd_steinberg( Mat input, Mat output )
{
    vector<Mat> channels;
    split(input, channels);
    for(int chan = 0; chan < channels.size(); chan++){

        Mat currentColorMat = channels[chan];
        currentColorMat.convertTo(currentColorMat, CV_32FC1, 1.0f/255.0f);

        for(int y = 0; y < currentColorMat.rows; y++) {
            for (int x = 0; x < currentColorMat.cols; x++) {
                float oldPixel = currentColorMat.at<float>(y, x);
                float newPixel = oldPixel > 0.5f ? 1.0f : 0.0f;
                float errorDelta = oldPixel - newPixel;

                currentColorMat.at<float>(y, x) = newPixel;

                if(x < input.cols - 1) {
                    currentColorMat.at<float>(y, x + 1) += (errorDelta * 7.0f/16.0f);
                }
                if(y < input.rows - 1){
                    if(x < input.cols - 1) {
                        currentColorMat.at<float>(y + 1, x + 1) += (errorDelta * 1.0f/16.0f);
                    }
                    if(x > 0) {
                        currentColorMat.at<float>(y + 1, x - 1) += (errorDelta * 3.0f/16.0f);
                    }
                    currentColorMat.at<float>(y + 1, x) += (errorDelta * 5.0f/16.0f);
                }
            }
        }
        channels[chan] = currentColorMat;
    }
    merge(channels, output);
    output.convertTo(output, input.type(), 0.0);
}

float distance_color_l2(Vec3f v1, Vec3f v2)
{
    return sqrt(pow(v1[0] - v2[0], 2) + pow(v1[1] - v2[1], 2) + pow(v1[2] - v2[2], 2));
}

int best_color(Vec3f reference, std::vector<Vec3f> colors)
{
    int nearest = 0;
    float distMin = std::numeric_limits<float>::max();
    for (int col = 0; col < colors.size(); col++) {
        float dist = distance_color_l2(reference, colors[col]);
        if (dist < distMin) {
            nearest = col;
            distMin = dist;
        }
    }
    return nearest;
}
Vec3f error_color( Vec3f bgr1, Vec3f bgr2 )
{
    return bgr1 - bgr2;
}


Mat tramage_floyd_steinberg( cv::Mat input, std::vector< cv::Vec3f > colors )
{
    cv::Mat fs;
    Vec3f c;
    int i;
    Vec3f e;
    input.convertTo( fs, CV_32FC3, 1/255.0);
    for(int y = 0; y < fs.rows; y++) {
        for (int x = 0; x < fs.cols; x++) {
            c = fs.at<Vec3f>(y, x);
            i = best_color(c, colors);
            e = error_color(c, colors[i]);
            fs.at<Vec3f>(y, x) = colors[i];

            if(x < input.cols - 1) {
                fs.at<Vec3f>(y, x + 1) += (e * 7.0f/16.0f);
            }
            if(y < input.rows - 1){
                if(x < input.cols - 1) {
                    fs.at<Vec3f>(y + 1, x + 1) += (e * 1.0f/16.0f);
                }
                if(x > 0) {
                    fs.at<Vec3f>(y + 1, x - 1) += (e * 3.0f/16.0f);
                }
                fs.at<Vec3f>(y + 1, x) += (e * 5.0f/16.0f);
            }
        }
    }
    Mat output;
    fs.convertTo( output, CV_8UC3, 255.0 );
    return output;
}

void afficheHistogramme(Mat image, string windowName){
    vector<double> hist = histogramme(image);
    vector<double> histCumul = histogramme_cumule(hist);
    Mat histImage = Mat(256, 512, CV_8UC1, Scalar(255, 255, 255));
    for (int i = 0; i < hist.size(); i++) {
        int histValue = (int)(hist[i]*255);
        for (int j = 0; j < histValue; j++) {
            histImage.at<uchar>(histImage.rows - 1 - j,i) = 0;
        }
        int histCumulValue = (int)(histCumul[i]*255);
        for (int j = 0; j < histCumulValue; j++) {
            histImage.at<uchar>(histImage.rows - 1 - j, i + hist.size()) = 0;
        }
    }
    namedWindow(windowName);
    imshow(windowName, histImage);
}

void afficheEgalisation(Mat frame)
{
    cvtColor(frame, frame, COLOR_BGR2HSV);
    vector<Mat> HSV;
    split(frame, HSV);
    afficheHistogramme(HSV[2], "Avant egalisation");
    egalisation(HSV[2]);
    afficheHistogramme(HSV[2], "Apres egalisation");
    merge(HSV, frame);
    cvtColor(frame, frame, cv::COLOR_HSV2BGR);

    namedWindow("Egalisation", WINDOW_AUTOSIZE);
    imshow("Egalisation", frame);
}

void afficheTramage1(Mat frame)
{
    Mat result;
    if(frame.channels() > 1){
        result = Mat(frame.rows, frame.cols, CV_32FC3);
    } else {
        result = Mat(frame.rows, frame.cols, CV_32FC1);
    }
    tramage_floyd_steinberg(frame, result);

    namedWindow("Tramage 1", WINDOW_AUTOSIZE);
    imshow("Tramage 1", result);
}

void afficheTramage2(Mat frame)
{
    if (frame.channels() < 3) {
        return;
    }

    vector<Vec3f> vects = {
            Vec3f({1.0, 0.0, 0.0}),
            Vec3f({1.0, 1.0, 0.0}),
            Vec3f({1.0, 1.0, 1.0}),
            Vec3f({0.0, 1.0, 1.0}),
            Vec3f({0.0, 1.0, 0.0}),
            Vec3f({0.0, 0.0, 1.0}),
            Vec3f({0.0, 0.0, 0.0}),
            Vec3f({1.0, 1.0, 1.0})
    };

    frame = tramage_floyd_steinberg(frame, vects);
    namedWindow("Tramage 2", WINDOW_AUTOSIZE);
    imshow("Tramage 2", frame);
}



int main(int, char**)
{
    VideoCapture cap(0);
    if(!cap.isOpened()) return -1;
    Mat frame, edges;
    for(;;)
    {
        cap >> frame;

        namedWindow("Input", WINDOW_AUTOSIZE);
        imshow("Input", frame);

        afficheTramage1(frame.clone());
        afficheEgalisation(frame);
        afficheTramage2(frame.clone());

        int   key_code = waitKey(30);
        int ascii_code = key_code & 0xff;
        if( ascii_code == 'q') break;
    }
    return 0;
}
