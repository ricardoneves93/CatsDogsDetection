#include <iostream>
#include <string>
#include <sys/stat.h>
#include <fstream>
#include "Opencv.h"

using namespace cv;
using namespace std;

const int CATS_CLASS = 0;
const int DOGS_CLASS = 1;
const int minHessian = 400;
const int numClusters = 100;
const int numTrainImages = 1000; //12499
const int numTestImages = 20; //12500

const string test_dir = "test\\";
const string dataset_dir = "train\\";

const string matcher_type = "BruteForce";
const string extractor_type = "SURF";
const string detector_type = "SURF";

Mat vocabulary;

string toLowerString(string str) {
	for (int i = 0; i < str.size(); i++) {
		str[i] = tolower(str[i]);
	}
	return str;
}

int getClass(string filepath) {
	string folder = filepath.substr(0, 5);
	folder = toLowerString(folder);

	int num_Chars = -1;
	if (folder == "train")
		num_Chars = 6;
	else
		num_Chars = 5;

	string classIdentifier = filepath.substr(num_Chars, 3);
	classIdentifier = toLowerString(classIdentifier);

	if (classIdentifier == "dog")
		return DOGS_CLASS;
	else if (classIdentifier == "cat")
		return CATS_CLASS;
	return -1;
}

inline bool fileExists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

vector<string> getFilesInDir() {
	vector <string> trainImages;
	string dogImgPath;
	string carImgPath;
	stringstream ss;
	for (int i = 0; i < numTrainImages; i++) {
		ss.str(string()); // Cleaning up stringstream
		ss << "dog." << i << ".jpg";
		dogImgPath = ss.str();
		trainImages.push_back(dogImgPath);

		ss.str(string()); // Cleaning up stringstream
		ss << "cat." << i << ".jpg";
		carImgPath = ss.str();
		trainImages.push_back(carImgPath);
	}

	return trainImages;
}

void createTrainingDescriptors(vector<string> trainImages) {

	//Mat img;
	//string filepath;
	SurfFeatureDetector detector(minHessian);

	SurfDescriptorExtractor extractor;
	//Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(extractor_type);
	
	
	Mat training_descriptors;
	/* Extracting features from training set thar contains all classes */
	for (int i = 0; i < trainImages.size(); i++) {
		string filepath = dataset_dir + trainImages[i];
		if (!fileExists(filepath)) {
			cout << "Error opening file! " << endl;
			cout << "Error in: " << filepath << endl;
			continue;
		}

		vector<KeyPoint> keypoints;
		Mat img = imread(filepath, CV_LOAD_IMAGE_COLOR);
		detector.detect(img, keypoints);

		Mat descriptors;
		extractor.compute(img, keypoints, descriptors);
		training_descriptors.push_back(descriptors);

		if (i % 500 == 0)
			cout << "Iteration: " << i << " of " << trainImages.size() << endl;
	}

	cout << "Finished training descriptors" << endl;

	// Creates vocabulary
	cout << "Going to do vocabulary" << endl;
	/* Creating the vocabulary (bag of words) */
	BOWKMeansTrainer bowtrainer(numClusters);
	bowtrainer.add(training_descriptors);
	vocabulary = bowtrainer.cluster();

	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << vocabulary;
	fs.release();

	cout << "Finished vocabulary creation" << endl;

}

/* Create histograms for each class (dog and cat) */
void computeClassesHistograms(vector <string> trainImages, vector<int>* classIdentifiers, map<int, Mat>* classes_training_dataset) {

	Ptr<FeatureDetector> detector = FeatureDetector::create(detector_type);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcher_type);
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(extractor_type);
	
	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	#pragma omp parallel for schedule(dynamic,3)
	for (int i = 0; i < trainImages.size(); i++) {

		string filepath = dataset_dir + trainImages[i];
		if (!fileExists(filepath)) {
			cout << "Error opening file! " << endl;
			cout << "Error in: " << filepath << endl;
			continue;
		}

		Mat currentImg = imread(filepath, CV_LOAD_IMAGE_COLOR);

		vector<KeyPoint> keypoints;
		detector->detect(currentImg, keypoints);

		Mat response_hist;
		bowide.compute(currentImg, keypoints, response_hist);

		int classIdentifier = getClass(filepath);
		
		#pragma omp critical 
		{
			if (classes_training_dataset->count(classIdentifier) == 0) {
				classes_training_dataset->operator[](classIdentifier).create(0, response_hist.cols, response_hist.type());
				classIdentifiers->push_back(classIdentifier);
			}
			classes_training_dataset->operator[](classIdentifier).push_back(response_hist);
		}
	}

	cout << "Finished creating histograms for both classes" << endl;
}

/* Bayes classification */
void bayesClassifier(vector<int>* classesNames, map<int, Mat>* classes_training_dataset, CvNormalBayesClassifier* bayes) {
	Mat samples;
	Mat labels;
	for (int i = 0; i < classesNames->size(); i++) {
		int classIdentifier = classesNames->at(i);
		Mat hist = classes_training_dataset->operator[](classIdentifier);

		//copy class samples and label
		samples.push_back(hist);
		labels.push_back( Mat(hist.rows, 1, hist.type(), Scalar(classIdentifier)) );
	}

	bayes->train(samples, labels);
	bayes->save("Models/model.bayes");

	cout << "Finished classifier" << endl;
}

void getTestImgs(vector <string>* testImagesPath) {
	string testImg;
	stringstream ss;
	for (int i = 1; i <= numTestImages; i++) {
		ss.str(string()); // Cleaning up stringstream
		ss << test_dir << i << ".jpg";
		testImg = ss.str();
		testImagesPath->push_back(testImg);
	}
}

/* Processe test images */
void testImages(vector <string>* testImagesPath, CvNormalBayesClassifier* bayes) {


	SurfFeatureDetector detector(minHessian);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcher_type);
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(extractor_type);

	BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	ofstream outputFile("results.csv");
	stringstream output;
	output << "id,label\n";
	int classID = -1;
	vector<KeyPoint> keypoints;

	for (int i = 0; i < testImagesPath->size(); i++) {
		classID = -1;
		Mat hist;
		keypoints.clear();
		Mat image = imread(testImagesPath->at(i), IMREAD_COLOR);
		detector.detect(image, keypoints);
		bowide.compute(image, keypoints, hist);

		if (hist.rows == 0) { //When detection goes wrong
			cout << "Merda" << endl;
			classID = 0;
		}
		else {
			classID = (int)bayes->predict(hist);
		}

		output << (i + 1) << "," << classID << "\n";
	}
	cout << output.str();
	outputFile << output.str();
	outputFile.close();
}

int main( int argc, char** argv ) {
	
	vector<string> trainImages = getFilesInDir();

	createTrainingDescriptors(trainImages);
	
	vector<int> classesNames;
	map<int, Mat> classes_training_dataset; 
	computeClassesHistograms(trainImages, &classesNames, &classes_training_dataset);

	CvNormalBayesClassifier bayes;
	bayesClassifier(&classesNames, &classes_training_dataset, &bayes);

	vector <string> testImagesPath;
	getTestImgs(&testImagesPath);

	testImages(&testImagesPath, &bayes);

	return 0;
}