#include <iostream>
#include <string>
#include <sys/stat.h>
#include <fstream>
#include "Opencv.h"

using namespace cv;
using namespace std;

const int CATS_CLASS = 0;
const int DOGS_CLASS = 1;

vector<string> getFilesInDir();
inline bool fileExists(const std::string&);
Mat createVocabulary(Mat);
Mat createTrainingDescriptors(string, vector<string>);
int getClass(string);
string toLowerString(string);
map<int, Mat> computeClassesHistograms(string, vector<string>, Mat, vector<int>*);
CvNormalBayesClassifier bayesClassifier(vector<int>*, map<int, Mat>);
void testImages(vector <string>*, Mat, CvNormalBayesClassifier*);
vector<string> getTestImgs(string);

int main( int argc, char** argv ) {
	
	string test_dir = "test\\";
	string dataset_dir = "train\\";
	vector <string> trainImages = getFilesInDir();

	Mat training_descriptors = createTrainingDescriptors(dataset_dir, trainImages);
	Mat vocabulary = createVocabulary(training_descriptors);
	
	vector<int> classesNames;
	map<int, Mat> classes_training_dataset; classes_training_dataset.clear();
	classes_training_dataset = computeClassesHistograms(dataset_dir, trainImages, vocabulary, &classesNames);

	CvNormalBayesClassifier bayes = bayesClassifier(&classesNames, classes_training_dataset);

	vector <string> testImagesPath = getTestImgs(test_dir);
	testImages(&testImagesPath, vocabulary, &bayes);

	return 0;
}

/* Processe test images */
void testImages(vector <string>* testImagesPath, Mat vocabulary, CvNormalBayesClassifier* bayes) {

	SurfFeatureDetector* detector = new SurfFeatureDetector();
	DescriptorMatcher* matcher = new BFMatcher(NORM_L2);
	OpponentColorDescriptorExtractor* extractor = new OpponentColorDescriptorExtractor(new SurfDescriptorExtractor());
	BOWImgDescriptorExtractor* bowide = new BOWImgDescriptorExtractor(extractor, matcher);
	bowide->setVocabulary(vocabulary);

	ofstream outputFile("results.csv");
	stringstream output; 
	output << "Img,Class\n";
	int classID = -1;
	vector<KeyPoint> keypoints;

	for (int i = 0; i < testImagesPath->size(); i++) {
		classID = -1;
		Mat hist;
		keypoints.clear();
		Mat image = imread(testImagesPath->at(i), IMREAD_COLOR);
		detector->detect(image, keypoints);
		bowide->compute(image, keypoints, hist);

		if (hist.rows == 0) { //When detection goes wrong
			cout << "Merda" << endl;
			classID = 0;
		}
		else {
			classID = (int) bayes->predict(hist);
		}

		string className = "";
		if (classID == DOGS_CLASS)
			className = "Dog";
		else if (classID == CATS_CLASS)
			className = "Cat";

		output << i << "," << className << "\n";
	}

	cout << output.str();
	outputFile << output.str();
}


/* Bayes classification */
CvNormalBayesClassifier bayesClassifier(vector<int>* classesNames, map<int, Mat> classes_training_dataset) {
	Mat samples;
	Mat labels;
	for (int i = 0; i < classesNames->size(); i++) {
		int classIdentifier = classesNames->at(i);
		Mat hist = classes_training_dataset[classIdentifier];

		//copy class samples and label
		samples.push_back(hist);
		Mat class_label(hist.rows, 1, hist.type(), Scalar(classIdentifier));
		labels.push_back(class_label);
	}

	CvNormalBayesClassifier bayes;
	bayes.train(samples, labels);
	bayes.save("Models/model.bayes");

	cout << "Finished classifier" << endl;

	return bayes;
}

/* Create histograms for each class (dog and cat) */
map<int, Mat> computeClassesHistograms(string dataset_dir, vector<string> trainImages, Mat vocabulary, vector<int>* classIdentifiers) {
	Mat currentImg;
	Mat response_hist;
	string filepath;
	classIdentifiers->clear();
	map<int, Mat> classes_training_dataset;
	vector<KeyPoint> keypoints; keypoints.clear();

	SurfFeatureDetector* detector = new SurfFeatureDetector();
	DescriptorMatcher* matcher = new BFMatcher(NORM_L2);
	OpponentColorDescriptorExtractor* extractor = new OpponentColorDescriptorExtractor(new SurfDescriptorExtractor());
	BOWImgDescriptorExtractor* bowide = new BOWImgDescriptorExtractor(extractor, matcher);
	bowide->setVocabulary(vocabulary);

	#pragma omp parallel for schedule(dynamic,3)
	for (int i = 0; i < trainImages.size(); i++) {

		filepath = dataset_dir + trainImages[i];
		if (!fileExists(filepath)) {
			cout << "Error opening file! " << endl;
			cout << "Error in: " << filepath << endl;
			continue;
		}

		currentImg = imread(filepath, CV_LOAD_IMAGE_COLOR);
		detector->detect(currentImg, keypoints);
		bowide->compute(currentImg, keypoints, response_hist);

		int classIdentifier = getClass(filepath);
		#pragma omp critical 
		{
			if (classes_training_dataset.count(classIdentifier) == 0) { //Not created yet
				classes_training_dataset[classIdentifier].create(0, response_hist.cols, response_hist.type());
				classIdentifiers->push_back(classIdentifier);
			}
			classes_training_dataset[classIdentifier].push_back(response_hist);
		}
	}

	cout << "Finished creating histograms for both classes" << endl;
	return classes_training_dataset;
}

Mat createTrainingDescriptors(string dataset_dir, vector<string> trainImages) {
	
	Mat img;
	string filepath;

	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);

	// PTR cenas nao faço ideia que merda é esta, ou que é para fazer aqui //
	//SurfDescriptorExtractor* extractor = new SurfDescriptorExtractor();
	OpponentColorDescriptorExtractor* extractor = new OpponentColorDescriptorExtractor(new SurfDescriptorExtractor());

	/* Extracting features from training set thar contains all classes */
	Mat training_descriptors;
	for (int i = 0; i < trainImages.size(); i++) {

		filepath = dataset_dir + trainImages[i];
		if (!fileExists(filepath)) {
			cout << "Error opening file! " << endl;
			cout << "Error in: " << filepath << endl;
			continue;
		}

		vector<KeyPoint> keypoints; keypoints.clear();
		img = imread(filepath, CV_LOAD_IMAGE_COLOR);
		detector.detect(img, keypoints);

		Mat descriptors;
		extractor->compute(img, keypoints, descriptors);
		training_descriptors.push_back(descriptors);
	}

	cout << "Finished creating training descriptors" << endl;
	return training_descriptors;
}

Mat createVocabulary(Mat training_descriptors) {
	/* Creating a vocabulary (bag of words) */
	int num_clusters = 100;
	BOWKMeansTrainer bowtrainer(num_clusters);
	bowtrainer.add(training_descriptors);
	Mat vocabulary = bowtrainer.cluster();

	cout << "Finished vocabulary creation" << endl;
	return vocabulary;
}

vector<string> getFilesInDir() {
	vector<string> trainImages; trainImages.clear();	
	for (int i = 0; i < 1000; i++) { //12499
		stringstream ss;
		
		ss << "dog." << i << ".jpg";
		string dogImgPath = ss.str();
		trainImages.push_back(dogImgPath);

		ss.str(string()); // Cleaning up stringstream
		ss << "cat." << i << ".jpg";
		string carImgPath = ss.str();
		trainImages.push_back(carImgPath);
	}
	return trainImages;
}

vector<string> getTestImgs(string test_dir) {
	vector<string> testImages; testImages.clear();
	for (int i = 1; i <= 12500; i++) {
		stringstream ss;
		ss << test_dir << i << ".jpg";
		string testImg = ss.str();
		testImages.push_back(testImg);
	}
	return testImages;
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

string toLowerString(string str) {
	for (int i = 0; i < str.size(); i++) {
		str[i] = tolower(str[i]);
	}
	return str;
}

inline bool fileExists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}