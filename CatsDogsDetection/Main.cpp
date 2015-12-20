#include <iostream>
#include <string>
#include <sys/stat.h>
#include "Opencv.h"

using namespace cv;
using namespace std;

vector<string> getFilesInDir();
inline bool fileExists(const std::string&);
Mat createVocabulary(Mat);
Mat createTrainingDescriptors(string, vector<string>);
string getClass(string filepath);
string toLowerString(string str);

int main( int argc, char** argv ) {
	
	string test_dir = "test\\";
	string dataset_dir = "train\\";
	vector <string> trainImages = getFilesInDir();

	Mat training_descriptors = createTrainingDescriptors(dataset_dir, trainImages);
	Mat vocabulary = createVocabulary(training_descriptors);

	/* Create histograms for each class (dog and cat) */
	int total_samples = 0;
	Mat currentImg;
	Mat response_hist;
	string filepath;
	vector<string> classNames; classNames.clear();
	map<string, Mat> classes_training_dataset;
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

		string className = getClass(filepath);
		#pragma omp critical 
		{
			if (classes_training_dataset.count(className) == 0) { //Not created yet
				classes_training_dataset[className].create(0, response_hist.cols, response_hist.type());
				classNames.push_back(className);
			}
			classes_training_dataset[className].push_back(response_hist);
		}
		total_samples++;
	}

	return 0;
}

/* Fazer refactoring, nao inicializar variáveis dentro do ciclo */
Mat createTrainingDescriptors(string dataset_dir, vector<string> trainImages) {
	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);

	// PTR cenas nao faço ideia que merda é esta, ou que é para fazer aqui //
	//SurfDescriptorExtractor* extractor = new SurfDescriptorExtractor();
	OpponentColorDescriptorExtractor* extractor = new OpponentColorDescriptorExtractor(new SurfDescriptorExtractor());

	/* Extracting features from training set thar contains all classes */
	Mat training_descriptors;
	for (int i = 0; i < trainImages.size(); i++) {

		string filepath = dataset_dir + trainImages[i];
		if (!fileExists(filepath)) {
			cout << "Error opening file! " << endl;
			cout << "Error in: " << filepath << endl;
			continue;
		}

		vector<KeyPoint> keypoints; keypoints.clear();
		Mat descriptors;
		Mat img = imread(filepath, CV_LOAD_IMAGE_COLOR);
		//imshow("img" + i, img);

		//waitKey(0);

		detector.detect(img, keypoints);
		extractor->compute(img, keypoints, descriptors);
		training_descriptors.push_back(descriptors);
	}

	cout << "Finished creating training descriptors" << endl;

	return training_descriptors;
}

Mat createVocabulary(Mat training_descriptors) {
	/* Creating a vocabulary (bag of words) */
	int num_clusters = 1000;
	BOWKMeansTrainer bowtrainer(num_clusters);
	bowtrainer.add(training_descriptors);
	Mat vocabulary = bowtrainer.cluster();

	cout << "Finished vocabulary creation" << endl;

	return vocabulary;
}

vector<string> getFilesInDir() {
	vector<string> trainImages; trainImages.clear();	
	for (int i = 0; i < 50; i++) {
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

string getClass(string filepath) {
	string folder = filepath.substr(0, 5);
	folder = toLowerString(folder);

	int num_Chars = -1;
	if (folder == "train")
		num_Chars = 6;
	else
		num_Chars = 5;

	string className = filepath.substr(num_Chars, 3);
	className = toLowerString(className);

	return className;
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