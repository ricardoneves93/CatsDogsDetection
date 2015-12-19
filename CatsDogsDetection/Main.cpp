#include <iostream>
#include "Opencv.h"

using namespace cv;
using namespace std;

vector<string> getFilesInDir();

int main( int argc, char** argv ) {
	
	string test_dir = "test\\";
	string dataset_dir = "train\\";

	vector <string> trainImages = getFilesInDir();

	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);
	SurfDescriptorExtractor extractor;

	// PTR //

	/* Extracting features from training set thar contains all classes */
	Mat training_descriptors;
	for (int i = 0; i < trainImages.size(); i++) {
		string filepath = dataset_dir + trainImages[i];
		vector<KeyPoint> keypoints; keypoints.clear();
		Mat descriptors;
		Mat img = imread(filepath);
		detector.detect(img, keypoints);
		extractor.compute(img, keypoints, descriptors);
		training_descriptors.push_back(descriptors);
	}

	/* Creating a vocabulary (bag of words) */
	int num_clusters = 1000;
	BOWKMeansTrainer bowtrainer(num_clusters);
	bowtrainer.add(training_descriptors);
	Mat vocabulary = bowtrainer.cluster();


	return 0;
}

vector<string> getFilesInDir() {
	vector<string> trainImages; trainImages.clear();
	for (int i = 0; i < 12499; i++) {
		stringstream ss;
		
		ss << "dog." << i << ".jpg";
		string dogImgPath = ss.str();
		trainImages.push_back(dogImgPath);

		ss.clear();
		ss << "cat." << i << ".jpg";
		string carImgPath = ss.str();
		trainImages.push_back(carImgPath);
	}
	return trainImages;
}