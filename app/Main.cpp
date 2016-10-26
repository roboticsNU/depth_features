 
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h> 
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "Main.h"

#define VEC_SIZE 340
#define FEATURES_RECURSIVE_DEPTH 3

int quad = 0;
double getMean(IplImage *sum, int startH, int endH, int startW, int endW) {

	double A = 0, B = 0, C = 0, D = 0;

	if (startH - 2 >= 0 && startW - 2 > 0) {
		A = ((double *)(sum->imageData + sum->widthStep * (startH - 2)))[startW - 2];
		//mat.at<double>(startH - 2, startW - 2);
	}
	if (startH - 2 >= 0) {

		B = ((double *)(sum->imageData + sum->widthStep * (startH - 2)))[endW - 1];
		//B = mat.at<double>(startH - 2, endW - 1);
	}
	if (startW - 2 > 0) {

		D = ((double *)(sum->imageData + sum->widthStep * (endH - 1)))[startW - 2];
		//  D = mat.at<double>(endH - 1, startW - 2);
	}

	C = ((double *)(sum->imageData + sum->widthStep * (endH - 1)))[endW - 1];
	//C = mat.at<double>(endH - 1, endW - 1);

	double num = ((endH - startH) + 1) * (endW - startW + 1);

	return (C + A - B - D) / num;
}

double getStd(IplImage *sq, int startH, int endH, int startW, int endW, double mean) {
	double A = 0, B = 0, C = 0, D = 0;

	if (startH - 2 >= 0 && startW - 2 > 0) {
		A = ((double *)(sq->imageData + sq->widthStep * (startH - 2)))[startW - 2];
		//mat.at<double>(startH - 2, startW - 2);
	}
	if (startH - 2 >= 0) {

		B = ((double *)(sq->imageData + sq->widthStep * (startH - 2)))[endW - 1];
		//B = mat.at<double>(startH - 2, endW - 1);
	}
	if (startW - 2 > 0) {

		D = ((double *)(sq->imageData + sq->widthStep * (endH - 1)))[startW - 2];
		//  D = mat.at<double>(endH - 1, startW - 2);
	}

	C = ((double *)(sq->imageData + sq->widthStep * (endH - 1)))[endW - 1];
	//C = mat.at<double>(endH - 1, endW - 1); 

	double num = ((endH - startH) + 1) * (endW - startW + 1);
	double sum = C + A - B - D;
	return sqrt(sum / num - mean*mean);
}


void estimateFeaturesManual(int startH, int endH, int startW, int endW, int step, double *answer, cv::Mat im) {
	int midH = 0;
	int midW = 0; 

	double min = im.at<float>(startH - 1, startW - 1);
	double max = min;

	double mean = 0;
	double count = 0;
	for (int i = startH - 1; i < endH; ++i) {
		for (int j = startW - 1; j < endW; ++j) {
			double val = im.at<float>(i, j);
			mean+=val;
			count++;
		}
	}

	mean /= (double)count;
	double std = 0;
	count = 0;
	for (int i = startH - 1; i < endH; ++i) {
		for (int j = startW - 1; j < endW; ++j) {
			double val = im.at<float>(i, j);
			std += pow(val - mean, 2.0);
			count++;
		}
	}

	std = sqrt(std/(double)count);

	for (int i = startH - 1; i < endH; ++i) {
		for (int j = startW - 1; j < endW; ++j) {
			double val = im.at<float>(i, j);
			if (val > max) {
				max = val;
			}

			if (val < min) {
				min = val;
			}
		}
	}

	answer[quad * 4 + 0] = min;
	answer[quad * 4 + 1] = max;
	answer[quad * 4 + 2] = mean;
	answer[quad * 4 + 3] = std;

	midH = (startH + endH) / 2;
	midW = (startW + endW) / 2;

	if (step != FEATURES_RECURSIVE_DEPTH) {
		quad++;
		estimateFeaturesManual(startH, midH, startW, midW, step+1, answer, im);
		quad++;
		estimateFeaturesManual(startH, midH, midW, endW, step+1, answer, im);
		quad++;
		estimateFeaturesManual(midH, endH, startW, midW, step+1, answer, im);
		quad++;
		estimateFeaturesManual(midH, endH, midW, endW, step+1, answer, im);
	}

}

void estimateFeatures(IplImage *sum, IplImage *sq, int startH, int endH, int startW, int endW, int step, double *answer, cv::Mat im) {
	int midH = 0;
	int midW = 0;

	double mean = getMean(sum, startH, endH, startW, endW);
	double std = getStd(sq, startH, endH, startW, endW, mean);

	double min = im.at<float>(startH - 1, startW - 1);
	double max = min;

	for (int i = startH - 1; i < endH; ++i) {
		for (int j = startW - 1; j < endW; ++j) {
			double val = im.at<float>(i, j);
			if (val > max) {
				max = val;
			}

			if (val < min) {
				min = val;
			}
		}
	}

	answer[quad * 4 + 0] = mean;
	answer[quad * 4 + 1] = std;
	answer[quad * 4 + 2] = max;
	answer[quad * 4 + 3] = min;

	midH = (startH + endH) / 2;
	midW = (startW + endW) / 2;

	if (step != FEATURES_RECURSIVE_DEPTH) {
		quad++;
		estimateFeatures(sum, sq, startH, midH, startW, midW, step+1, answer, im);
		quad++;
		estimateFeatures(sum, sq, startH, midH, midW, endW, step+1, answer, im);
		quad++;
		estimateFeatures(sum, sq, midH, endH, startW, midW, step+1, answer, im);
		quad++;
		estimateFeatures(sum, sq, midH, endH, midW, endW, step+1, answer, im);
	}

}
double *getFeatureVector(IplImage *im) {
	IplImage *sum = cvCreateImage(cvSize(im->width + 1, im->height + 1), IPL_DEPTH_64F, 1);
	IplImage *sqsum = cvCreateImage(cvSize(im->width + 1, im->height + 1), IPL_DEPTH_64F, 1);

	cvZero(sum);
	cvZero(sqsum);

	cv::Mat mat = cv::Mat(im, true);
	cvIntegral(im, sum, sqsum);
	double *answer = (double *)malloc(sizeof(double) * (341) * 4);

	quad = 0;
	estimateFeatures(sum, sqsum, 1, 240, 1, 320, 0, answer, mat);

	cvReleaseImage(&sum);
	cvReleaseImage(&sqsum);
	return answer;
}

double *getFeatureVectorManual(IplImage *im) {
	cv::Mat mat = cv::Mat(im, true);
	double *answer = (double *)malloc(sizeof(double) * (341) * 4);

	quad = 0;
	estimateFeaturesManual(1, 240, 1, 320, 0, answer, mat); 
	return answer;
}

IplImage *filterImage(IplImage *conf, IplImage *depth) {
	// TODO
	int w = conf->width;
	int h = conf->height;  
	IplImage *img = cvCreateImage(cvGetSize(conf), IPL_DEPTH_32F, 1);
	cvZero(img);
	/*
	for (int ii = 0; ii < h; ++ii) {   
	for (int jj = 0; jj < w; ++jj) {
	ushort *depthPtr = (ushort*)(depth->imageData + ii * depth->widthStep);
	float *ptr = (float *)(img->imageData + ii * img->widthStep);
	ptr[jj] = depthPtr[jj];

	}
	}
	return img;*/
	for (int i = 0; i < h; ++i) {   
		for (int j = 0; j < w; ++j) {
			int yStart = 0, yEnd = 0, xStart = 0, xEnd = 0;
			yStart = i != 0 ? i - 1: 0;
			xStart = j != 0 ? j - 1: 0;

			yEnd = i != h - 1 ? i + 1:  h - 1;
			xEnd = j != w - 1 ? j + 1:  w - 1;

			double val = 0;
			double sum = 0;
			for (int ii = yStart; ii <= yEnd; ++ii) {
				ushort *confPtr = (ushort*)(conf->imageData + ii * conf->widthStep);
				ushort *depthPtr = (ushort*)(depth->imageData + ii * depth->widthStep);
				for (int jj = xStart; jj <= xEnd; ++jj) {
					double confVal = (double)confPtr[jj] / (double)0xFFFF;
					ushort depthVal = depthPtr[jj];
					if (depthVal > 2000) {
						depthVal = 2000;
					}

					val += confVal * depthVal;
					sum += confVal;
				}
			}
			val /= sum;

			float *ptr = (float *)(img->imageData + i * img->widthStep);
			ptr[j] = val;
		}
	}
	return img;
}

char *classNames[7] = {"run", "walk", "stand", "stairs up", "stairs down", "slope up", "slope down"};

void normalizeFeatures(double * feature) {
	double featMin = feature[0];
	double featMax = feature[0]; 

	for (int i = 0; i < VEC_SIZE; ++i) {
		if (feature[i] > featMax) {
			featMax = feature[i];
		}
		if (feature[i] < featMin) {
			featMin = feature[i];
		}
	}

	double diff = featMax - featMin;

	for (int i = 0; i < VEC_SIZE; ++i) {
		feature[i] -= featMin;
		feature[i] = 2.0 * feature[i] / diff - 1;
	}	
}
int startFrom = 10;
void generateFeaturesFor(int classLabel, const char *pathToFiles, int startInd, int endInd, FILE *file) {
	  
	for (int ind = startInd; ind <= endInd; ++ind) {
		char nameDepth[255];
		char nameDepthPrev[255];
		char nameConf[255];
		char nameConfPrev[255];
		char indexStr[30];
		char indexStrPrev[30];
		char indNumStr[5]; 

		char nameDepthPrev2[255];
		char nameConfPrev2[255];
		char indexStrPrev2[30];

		sprintf(indexStr, "%05d", ind);
		sprintf(indexStrPrev, "%05d", ind - startFrom / 2);
		sprintf(indexStrPrev2, "%05d", ind - startFrom);
		
		 
			strcpy(nameDepth, pathToFiles);
			strcpy(nameConf, pathToFiles);
			strcat(nameDepth, "depth\\depth");
			strcat(nameConf, "conf\\conf");

			strcpy(nameDepthPrev, pathToFiles);
			strcpy(nameConfPrev, pathToFiles);
			strcat(nameDepthPrev, "depth\\depth");
			strcat(nameConfPrev, "conf\\conf");
			
			strcpy(nameDepthPrev2, pathToFiles);
			strcpy(nameConfPrev2, pathToFiles);
			strcat(nameDepthPrev2, "depth\\depth");
			strcat(nameConfPrev2, "conf\\conf");

			strcat(nameDepth, indexStr);
			strcat(nameConf, indexStr);

			strcat(nameDepthPrev, indexStrPrev);
			strcat(nameConfPrev, indexStrPrev);

			strcat(nameDepthPrev2, indexStrPrev2);
			strcat(nameConfPrev2, indexStrPrev2);

			strcat(nameDepth, ".png");
			strcat(nameConf, ".png");

			strcat(nameDepthPrev, ".png");
			strcat(nameConfPrev, ".png");

			strcat(nameDepthPrev2, ".png");
			strcat(nameConfPrev2, ".png");

		IplImage *imDepth = cvLoadImage(nameDepth, CV_LOAD_IMAGE_ANYDEPTH );
		IplImage *imConf = cvLoadImage(nameConf, CV_LOAD_IMAGE_ANYDEPTH );
		IplImage *filtered = filterImage(imConf, imDepth);

		IplImage *imDepthPrev = cvLoadImage(nameDepthPrev, CV_LOAD_IMAGE_ANYDEPTH);
		IplImage *imConfPrev = cvLoadImage(nameConfPrev, CV_LOAD_IMAGE_ANYDEPTH);
		IplImage *filteredPrev = filterImage(imConfPrev, imDepthPrev);

		IplImage *imDepthPrev2 = cvLoadImage(nameDepthPrev2, CV_LOAD_IMAGE_ANYDEPTH);
		IplImage *imConfPrev2 = cvLoadImage(nameConfPrev2, CV_LOAD_IMAGE_ANYDEPTH);
		IplImage *filteredPrev2 = filterImage(imConfPrev2, imDepthPrev2);

		double * feature = getFeatureVectorManual(filtered);

		IplImage *subtracted = cvCloneImage(filtered);
		cvZero(subtracted);
		IplImage *subtracted2 = cvCloneImage(filtered);
		cvZero(subtracted2);

		// 10 - 5
		cvSub(filtered, filteredPrev, subtracted);
		// 5 - 1
		cvSub(filteredPrev, filteredPrev2, subtracted2);

		double *featuresPrev = getFeatureVectorManual(subtracted);
		double *featuresPrev2 = getFeatureVectorManual(subtracted2);

		/*normalizeFeatures(feature);
		normalizeFeatures(featuresPrev);*/

		cvReleaseImage(&imDepth);
		cvReleaseImage(&imConf);
		cvReleaseImage(&filtered);
		cvReleaseImage(&imDepthPrev);
		cvReleaseImage(&imConfPrev);
		cvReleaseImage(&filteredPrev);
		cvReleaseImage(&imDepthPrev2);
		cvReleaseImage(&imConfPrev2);
		cvReleaseImage(&filteredPrev2);
		cvReleaseImage(&subtracted);
		cvReleaseImage(&subtracted2);

		int labelNum = classLabel;
		if (classLabel >= 6) {
			labelNum = 2;
		}
		fprintf(file, "%d\t", labelNum);
		for (int i = 0; i < VEC_SIZE; ++i) {
			fprintf(file, "%f\t", feature[i]);
		}

		for (int i = 0; i < VEC_SIZE; ++i) {
			fprintf(file, "%f\t", featuresPrev[i]);
		}

		for (int i = 0; i < VEC_SIZE; ++i) {
			fprintf(file, "%f\t", featuresPrev2[i]);
		}
		fprintf(file, "\n");
		free(feature);
		free(featuresPrev);
		free(featuresPrev2);
	}
}

int  parseName(std::string name);

bool parseAnnotation(std::string annotation, int *classLabel, int *start, int *end) {
	int len = annotation.length();
	if (len <= 0) {
		return false;
	}
	std::string nameStr;
	int num1 = 0;
	int num2 = 0;
	int stage = 0;
	int st = 0;

	for (int i = 0; i < len; ++i) {
		char ch = annotation.at(i);
		if (ch == ' ' || ch == '\t') {
			continue;
		}
		else if (!isdigit(ch)) {
			nameStr += ch;
		}
		else {
			st = i;
			break;
		}
	}

	stage = 0;
	for (int i = st; i < len; ++i) {
		char ch = annotation.at(i);
		if (stage == 0 && (ch == ' ' || ch == '\t')) {
			continue;
		}
		else if (isdigit(ch)) {
			stage = 1;
			num1 = num1 * 10 + (int)(ch - '0');
		}
		else {
			st = i;
			break;
		}
	}

	stage = 0;
	for (int i = st; i < len; ++i) {
		char ch = annotation.at(i);
		if (stage == 0 && (ch == ' ' || ch == '\t')) {
			continue;
		}
		else if (isdigit(ch)) {
			stage = 1;
			num2 = num2 * 10 + (int)(ch - '0');
		}
		else {
			st = i;
			break;
		}
	}

	*start = num1;
	*end = num2;
	*classLabel = parseName(nameStr);
	return true;
}
 
#define RUN 1
#define WALK 2
#define STAND 3
#define STAIRSUP 4
#define STAIRSDOWN 5
#define SLOPEUP 6
#define SLOPEDOWN 7
  
void generateFeatures(std::string path) {

	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/
	
	const char *name = "features10-5_5-1";
	char buffer[3];
	std::string annotation;
	std::ifstream infile;
	infile.open(path + "annotations.txt");
	int a = 0;

	FILE *file = NULL;
	char *cstr = new char[path.length() + 1 + 12 + 10];
	strcpy(cstr, path.c_str());
	strcat(cstr, name); 
	strcat(cstr, ".txt");
	
	file = fopen(cstr, "w");

	while (!infile.eof()) {
		getline(infile, annotation);
		if (annotation.empty()) {
			continue;
		}
		 
		int classLabel = 0;
		int startInd = 0, endInd = 0;
		parseAnnotation(annotation, &classLabel, &startInd, &endInd);
		generateFeaturesFor(classLabel, path.c_str(), startInd, endInd, file);
	}
	infile.close();
	printf("GENERATION DONE FOR: %s\n", path.c_str());

	delete[] cstr;
} 

int parseName(std::string name) {
	if (name.compare("WALK") == 0) {
		return WALK;
	}

	if (name.compare("STAND") == 0) {
		return STAND;
	}

	if (name.compare("STAIRSUP") == 0) {
		return STAIRSUP;
	}

	if (name.compare("STAIRSDOWN") == 0) {
		return STAIRSDOWN;
	}
	if (name.compare("RUN") == 0) {
		return RUN;
	}

	assert(false);
}

void generateForSubject(std::string line) {
	std::string path;
	std::ifstream infile;
	infile.open(line + "readme.txt");
	int a = 0;
	while (!infile.eof()) {
		getline(infile, path);
		generateFeatures(path);
	}
	infile.close();
	printf("GENERATION DONE FOR: %s\n", line.c_str());
}

void processFile(char *name) {
	std::string line;
	std::ifstream infile;
	infile.open(name);
	int a = 0;
	while (!infile.eof()) {
		getline(infile, line);
		generateForSubject(line);
	}
	infile.close();
	std::cout << "Processing done for " << name;
}
int main(int argc, char **argv) { 
	processFile("Z:/Yerzhan/12people/paper_videos_data/depth/readme_first.txt");
}
