// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "Functions.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <queue>
#include <opencv2/core/utils/logger.hpp>


#define MAX_HUE 256
#define HISTOGRAM_THRESHOLD 0.1 //prag pt eliminare zgomote hist
#define BLUE_FUSION_THRESHOLD 5.0 //prag funzionare culori

Rect selectedROI;
bool selectingROI = false;

// structura pentru stocarea informatiilor despre modelul de culoare
struct ColorModel {
	int hist[MAX_HUE] = { 0 }; 
	float mean = 0;
	float stddev = 0;
};

//selectia manuala a ROI
void onMouseSelectROI(int event, int x, int y, int flags, void* param) {
	static Point startPoint;
	if (event == EVENT_LBUTTONDOWN) {
		selectingROI = true;
		startPoint = Point(x, y);
	}
	else if (event == EVENT_MOUSEMOVE && selectingROI) {
		Mat tempImage = *((Mat*)param);
		Mat clone = tempImage.clone();
		rectangle(clone, startPoint, Point(x, y), Scalar(0, 255, 0), 1);
		imshow("Select ROI", clone);
	}
	else if (event == EVENT_LBUTTONUP) {
		selectingROI = false;
		selectedROI = Rect(Point(startPoint.x, startPoint.y), Point(x, y));
	}
}

//adauga la histograma globala
void addToGlobalHistogram(const Mat& roi, ColorModel& model) {
	vector<Mat> channels;
	split(roi, channels);
	Mat hueChannel = channels[0];

	for (int y = 0; y < hueChannel.rows; ++y) {
		for (int x = 0; x < hueChannel.cols; ++x) {
			int hue = hueChannel.at<uchar>(y, x);
			model.hist[hue]++;
		}
	}
}

// filtreaza histograma globala cu un filtru Gaussian
void filterHistogram(ColorModel& model) {
	float gauss[7];
	float sigma = 1.5, sqrt2pi = sqrtf(2 * CV_PI), sum = 0;

	//construire filtru Gaussian 1D
	for (int i = 0; i < 7; i++) {
		gauss[i] = (1.0 / (sqrt2pi * sigma)) * exp(-((i - 3) * (i - 3)) / (2 * sigma * sigma));
		sum += gauss[i];
	}

	//normalizare filtru Gaussian
	for (int i = 0; i < 7; i++) gauss[i] /= sum;

	//aplicare filtru Gaussian pe histograma
	int filteredHist[MAX_HUE] = { 0 };
	for (int j = 3; j < MAX_HUE - 3; ++j) {
		for (int i = 0; i < 7; ++i) {
			filteredHist[j] += (int)(model.hist[j + i - 3] * gauss[i]);
		}
	}

	memcpy(model.hist, filteredHist, sizeof(filteredHist));
}

//elimina valorile mici din histograma (zgomot)
void removeNoiseFromHistogram(ColorModel& model) {
	int max_value = *max_element(model.hist, model.hist + MAX_HUE);
	int threshold = max_value * HISTOGRAM_THRESHOLD;

	for (int j = 0; j < MAX_HUE; ++j) {
		if (model.hist[j] < threshold) {
			model.hist[j] = 0;
		}
	}
}

//calculeaza media si deviatia standard pe baza histogramelor
void calculateMeanAndStdDev(ColorModel& model) {
	float total = 0;
	model.mean = 0;
	model.stddev = 0;

	for (int i = 0; i < MAX_HUE; ++i) {
		total += model.hist[i];
		model.mean += i * model.hist[i];
	}
	model.mean /= total;

	for (int i = 0; i < MAX_HUE; ++i) {
		model.stddev += (i - model.mean) * (i - model.mean) * model.hist[i];
	}
	model.stddev = sqrt(model.stddev / total);
}

//construieste modelul de culoare pentru o culoare data
void trainColorModel(const Mat& hsv, ColorModel& model) {
	Mat displayImage = hsv.clone();
	namedWindow("Select ROI");
	setMouseCallback("Select ROI", onMouseSelectROI, &displayImage);
	imshow("Select ROI", displayImage);

	cout << "Selecteaza o regiune cu mouse-ul si apasa Enter pentru a confirma." << endl;
	waitKey(0);
	destroyWindow("Select ROI");

	if (selectedROI.area() == 0) {
		cerr << "Nu a fost selectata nicio regiune valida!" << endl;
		return;
	}

	Mat roi = hsv(selectedROI);
	addToGlobalHistogram(roi, model);
	filterHistogram(model);
	removeNoiseFromHistogram(model);
	calculateMeanAndStdDev(model);

	cout << "Model calculat: Mean = " << model.mean << ", StdDev = " << model.stddev << endl;
}



//segmentare pe baza modelului de culoare
void segmentImage(const Mat& hsv, const ColorModel& model, Mat& mask, float k = 2.5) {
	Mat blurred;
	GaussianBlur(hsv, blurred, Size(5, 5), 0, 0);

	int minHue = max(0, int(model.mean - k * model.stddev));
	int maxHue = min(MAX_HUE - 1, int(model.mean + k * model.stddev));

	inRange(blurred, Scalar(minHue, 50, 50), Scalar(maxHue, 255, 255), mask);

	//postprocesare morfologica
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(mask, mask, element, Point(-1, -1), 2);
	dilate(mask, mask, element, Point(-1, -1), 2);

	//filtrare componente conexe
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area < 300) { 
			drawContours(mask, contours, i, Scalar(0), FILLED);
		}
	}
}


//fuzioneaza cele doua modele de culoare
ColorModel mergeColorModels(const ColorModel& model1, const ColorModel& model2) {
	ColorModel mergedModel;
	mergedModel.mean = (model1.mean + model2.mean) / 2;
	mergedModel.stddev = max(model1.stddev, model2.stddev);

	for (int i = 0; i < MAX_HUE; ++i) {
		mergedModel.hist[i] = model1.hist[i] + model2.hist[i];
	}

	return mergedModel;
}

//proceseaza modelele pentru nuantele de albastru
void processBlueModels(const Mat& hsv, const ColorModel& blue1Model, const ColorModel& blue2Model, Mat& maskBlue) {
	if (abs(blue1Model.mean - blue2Model.mean) < BLUE_FUSION_THRESHOLD) {
		cout << "Fuzionam cele doua modele de albastru." << endl;
		ColorModel combinedBlueModel = mergeColorModels(blue1Model, blue2Model);
		segmentImage(hsv, combinedBlueModel, maskBlue);
	}
	else {
		cout << "Folosim modele separate pentru cele doua nuante de albastru." << endl;
		Mat maskBlue1, maskBlue2;
		segmentImage(hsv, blue1Model, maskBlue1);
		segmentImage(hsv, blue2Model, maskBlue2);
		maskBlue = maskBlue1 | maskBlue2;
	}
}


//numara degetele detectate pe baza contururilor
int countFingersByMask(const Mat& mask) {
	vector<vector<Point>> contours;
	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	int count = 0;
	for (const auto& contour : contours) {
		if (contourArea(contour) > 300) {
			count++;
		}
	}
	return count;
}

//functie principala pentru antrenare si testare
int test() {
	char fname[MAX_PATH];
	vector<string> trainingImages;

	cout << "Selecteaza imaginile pentru antrenare." << endl;
	while (openFileDlg(fname)) {
		trainingImages.push_back(fname);
	}

	ColorModel redModel, greenModel, yellowModel, blue1Model, blue2Model;

	for (const auto& imagePath : trainingImages) {
		Mat frame = imread(imagePath);
		if (frame.empty()) {
			cerr << "Nu pot deschide imaginea: " << imagePath << endl;
			continue;
		}

		Mat hsv;
		cvtColor(frame,hsv, COLOR_BGR2HSV);

		cout << "Regiunea pentru rosu..." << endl;
		trainColorModel(hsv, redModel);

		cout << "Regiunea pentru verde..." << endl;
		trainColorModel(hsv, greenModel);

		cout << "Regiunea pentru galben..." << endl;
		trainColorModel(hsv, yellowModel);

		cout << "Regiunea pentru prima nuanta de albastru..." << endl;
		trainColorModel(hsv, blue1Model);

		cout << "Regiunea pentru a doua nuanta de albastru..." << endl;
		trainColorModel(hsv, blue2Model);
	}

	cout << "Selecteaza imaginile pentru testare." << endl;
	while (openFileDlg(fname)) {
		Mat testFrame = imread(fname);
		if (testFrame.empty()) {
			cerr << "Nu pot deschide imaginea pentru testare!" << endl;
			continue;
		}

		Mat hsv;
		cvtColor(testFrame, hsv, COLOR_BGR2HSV);

		Mat maskRed, maskGreen, maskYellow, maskBlue;
		segmentImage(hsv, redModel, maskRed);
		segmentImage(hsv, greenModel, maskGreen);
		segmentImage(hsv, yellowModel, maskYellow);
		processBlueModels(hsv, blue1Model, blue2Model, maskBlue);

		imshow("Masca Rosu", maskRed);
		imshow("Masca Verde", maskGreen);
		imshow("Masca Galben", maskYellow);
		imshow("Masca Albastru", maskBlue);

		int redFingers = countFingersByMask(maskRed);
		int greenFingers = countFingersByMask(maskGreen);
		int yellowFingers = countFingersByMask(maskYellow);
		int blueFingers = countFingersByMask(maskBlue);

		int totalFingers = redFingers + greenFingers + yellowFingers + blueFingers;

		cout << "Nr degete detectate: " << endl;
		cout << "Rosu: " << redFingers << endl;
		cout << "Verde: " << greenFingers << endl;
		cout << "Galben: " << yellowFingers << endl;
		cout << "Albastru: " << blueFingers << endl;
		cout << "Total degete: " << totalFingers << endl;

		Mat combined = maskRed | maskGreen | maskYellow | maskBlue;
		putText(combined, "Total Fingers: " + to_string(totalFingers), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255), 2);
		imshow("Segmentare finala", combined);

		waitKey(0);
	}

	return 0;
}


int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf("1 - PROIECT\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				test();
				break;
		}
	}
	while (op!=0);
	return 0;
}