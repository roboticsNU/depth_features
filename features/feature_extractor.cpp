#include "feature_extractor.h"


#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui_c.h" 
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/videoio.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2\core\core.hpp>
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
int startFrom = 5;
void generateFeaturesFor(int trainset, int classLabel, char *pathToFiles, char *filenamePrefix, int startInd, int endInd, FILE *file) {
	  
	for (int ind = startInd; ind <= endInd; ++ind) {
		char nameDepth[255];
		char nameDepthPrev[255];
		char nameConf[255];
		char nameConfPrev[255];
		char indexStr[30];
		char indexStrPrev[30];
		char indNumStr[5]; 

		sprintf(indexStr, "%05d", ind);
		sprintf(indexStrPrev, "%05d", ind - startFrom);
		
		 
			strcpy(nameDepth, pathToFiles);
			strcpy(nameConf, pathToFiles);
			strcat(nameDepth, "depth\\depth");
			strcat(nameConf, "conf\\conf");

			strcpy(nameDepthPrev, pathToFiles);
			strcpy(nameConfPrev, pathToFiles);
			strcat(nameDepthPrev, "depth\\depth");
			strcat(nameConfPrev, "conf\\conf");
			
			strcat(nameDepth, indexStr);
			strcat(nameConf, indexStr);

			strcat(nameDepthPrev, indexStrPrev);
			strcat(nameConfPrev, indexStrPrev);

			strcat(nameDepth, ".png");
			strcat(nameConf, ".png");

			strcat(nameDepthPrev, ".png");
			strcat(nameConfPrev, ".png"); 

		IplImage *imDepth = cvLoadImage(nameDepth, CV_LOAD_IMAGE_ANYDEPTH );
		IplImage *imConf = cvLoadImage(nameConf, CV_LOAD_IMAGE_ANYDEPTH );
		IplImage *filtered = filterImage(imConf, imDepth);

		IplImage *imDepthPrev = cvLoadImage(nameDepthPrev, CV_LOAD_IMAGE_ANYDEPTH );
		IplImage *imConfPrev = cvLoadImage(nameConfPrev, CV_LOAD_IMAGE_ANYDEPTH );
		IplImage *filteredPrev = filterImage(imConfPrev, imDepthPrev);

		double * feature = getFeatureVectorManual(filtered);

		IplImage *subtracted = cvCloneImage(filtered);
		cvZero(subtracted);

		cvSub(filtered, filteredPrev, subtracted);

		double *featuresPrev = getFeatureVectorManual(subtracted);

		/*normalizeFeatures(feature);
		normalizeFeatures(featuresPrev);*/

		cvReleaseImage(&imDepth);
		cvReleaseImage(&imConf);
		cvReleaseImage(&filtered);
		cvReleaseImage(&imDepthPrev);
		cvReleaseImage(&imConfPrev);
		cvReleaseImage(&filteredPrev);
		cvReleaseImage(&subtracted);

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
		fprintf(file, "\n");
		free(feature);
		free(featuresPrev);
	}
}

#define PATH_TEST_4 "C:\\robotics\\dataset4\\data\\depthsense\\"
#define PATH_TEST_5 "C:\\robotics\\dataset5\\data\\depthsense\\"
#define PATH_TEST_6 "C:\\robotics\\dataset6\\data\\depthsense\\"
#define PATH_TEST_7 "C:\\robotics\\dataset7\\data\\depthsense\\"
#define PATH_TEST_8 "C:\\robotics\\dataset8\\depthsense\\"
#define PATH_TEST_9 "C:\\robotics\\dataset9\\data\\depthsense\\"


#define PATH_SET_1 "C:\\robotics\\dataset1\\data\\depthsense\\"
#define PATH_SET_2 "C:\\robotics\\dataset2\\data\\depthsense\\"
#define PATH_SET_3 "C:\\robotics\\dataset3\\data\\depthsense\\"
#define PATH_SET_4 "C:\\robotics\\dataset4\\data\\depthsense\\"
#define PATH_SET_5 "C:\\robotics\\dataset5\\data\\depthsense\\"
#define PATH_SET_6 "C:\\robotics\\dataset6\\data\\depthsense\\"
#define PATH_SET_7 "C:\\robotics\\dataset7\\data\\depthsense\\"
#define PATH_SET_8 "C:\\robotics\\dataset8\\data\\depthsense\\"
#define PATH_SET_9 "C:\\robotics\\dataset9\\data\\depthsense\\"

#define PATH_TO_FILES "C:\\robotics\\training\\"
#define RUN 1
#define WALK 2
#define STAND 3
#define STAIRSUP 4
#define STAIRSDOWN 5
#define SLOPEUP 6
#define SLOPEDOWN 7

void train1(FILE *file) {
	// TRAINING 1 

	// RUN 1, TRAIN 1
	generateFeaturesFor(1, RUN, PATH_TO_FILES, "depth", 1661, 1800, file);	// 139
	//generateFeaturesFor(1, RUN, PATH_TO_FILES, "depth", 1661, 1661 + 131, file);	// 139 remove it
	printf("training 1 run done\n"); // 139

	// WALK 2, TRAIN 1
	generateFeaturesFor(1, 2, PATH_TO_FILES, "depth", 690, 900, file);		// 210
	generateFeaturesFor(1, 2, PATH_TO_FILES, "depth", 1165,	1600, file);	// 435
	generateFeaturesFor(1, 2, PATH_TO_FILES, "depth", 2065,	2700, file);	// 635
	generateFeaturesFor(1, 2, PATH_TO_FILES, "depth", 3022,	3181, file);	// 159
	
	generateFeaturesFor(1, WALK, PATH_TO_FILES, "depth", 4350,	4450, file);	// 100
	generateFeaturesFor(1, 2, PATH_TO_FILES, "depth", 4620,	4700, file);	// 80
	generateFeaturesFor(1, 2, PATH_TO_FILES, "depth", 5650,	5850, file);	// 200
	printf("training 1 walk done\n"); // 210+435+635+159+100+280=1819

	// STAND 3, TRAIN 1
	generateFeaturesFor(1, 3, PATH_TO_FILES, "depth", 1885,	2060, file);	// 175
	generateFeaturesFor(1, STAND, PATH_TO_FILES, "depth", 4500,	4600, file);	// 100
	printf("training 1 stand done\n"); // 275

	// STAIRS up 4, TRAIN 1
	generateFeaturesFor(1, STAIRSUP, PATH_TO_FILES, "depth", 2814,	3000, file);	// 186
	
	printf("training 1 stairs up done\n"); // 186
	// STAIRS down 5, TRAIN 1
	generateFeaturesFor(1, STAIRSDOWN, PATH_TO_FILES, "depth", 920,	1118, file);	// 198
	printf("training 1 stairs down done\n"); // 198

	// SLOPE UP 6, TRAIN 1	
	//generateFeaturesFor(1, 6, PATH_TO_FILES, "depth", 3210,	3307, file);	// 97
	//generateFeaturesFor(1, SLOPEUP, PATH_TO_FILES, "depth", 4740,	4900, file);	// 160
	//generateFeaturesFor(1, 6, PATH_TO_FILES, "depth", 5000,	5170, file);	// 170

	printf("training 1 slope up done\n"); // 427

	// SLOPE DOWN 7 
	//generateFeaturesFor(1, 7, PATH_TO_FILES, "depth", 3585,	3675, file);	// 90
	//generateFeaturesFor(1, 7, PATH_TO_FILES, "depth", 3885,	4100, file);	// 215
	//generateFeaturesFor(1, SLOPEDOWN, PATH_TO_FILES, "depth", 4255,	4315, file);	// 60
	//generateFeaturesFor(1, SLOPEDOWN, PATH_TO_FILES, "depth", 5545,	5620, file);	// 75

	printf("training 1 slope down done\n"); // 440  
}

void train2(FILE *file) {
// TRAINING 2 
	//1: RUN 
	generateFeaturesFor(2, 1, PATH_TO_FILES, "dep-conf_", 3220, 3460, file);	// 240
	printf("training 2 run done\n"); // 240

	//2: WALK
	generateFeaturesFor(2, 2, PATH_TO_FILES, "dep-conf_", 1,	271, file);		// 271
	generateFeaturesFor(2, 2, PATH_TO_FILES, "dep-conf_", 370,	564, file);		// 194
	generateFeaturesFor(2, 2, PATH_TO_FILES, "dep-conf_", 684,	724, file);		// 40
	generateFeaturesFor(2, WALK, PATH_TO_FILES, "dep-conf_", 1193,	1360, file);	// 167
	generateFeaturesFor(2, 2, PATH_TO_FILES, "dep-conf_", 1521,	1680, file);	// 159
	generateFeaturesFor(2, 2, PATH_TO_FILES, "dep-conf_", 2190,	2252, file);	// 62
	generateFeaturesFor(2, 2, PATH_TO_FILES, "dep-conf_", 2674,	2820, file);	// 146
	generateFeaturesFor(2, 2, PATH_TO_FILES, "dep-conf_", 3087,	3186, file);	// 99
	generateFeaturesFor(2, 2, PATH_TO_FILES, "dep-conf_", 3580,	4610, file);	// 1030
	generateFeaturesFor(2, 2, PATH_TO_FILES, "dep-conf_", 4890,	5060, file);	// 170
	printf("training 2 walk done\n"); // 271+194+40+167+159+62+146+99+1030+170=2338

	//3: STAND
	generateFeaturesFor(2, STAND, PATH_TO_FILES, "dep-conf_", 1369,	1512, file); // 143
	printf("training 2 stand done\n"); // 143

	//4: stairs up
	generateFeaturesFor(2, STAIRSUP, PATH_TO_FILES, "dep-conf_", 4640,	4866, file);	// 226
	printf("training 2 stairs up done\n"); // 226

	//5: stairs down
	generateFeaturesFor(2, STAIRSDOWN, PATH_TO_FILES, "dep-conf_", 2850,	3060, file);	// 210
	printf("training 2 stairs down done\n"); // 210

	//6: slope up 
	//generateFeaturesFor(2, SLOPEUP, PATH_TO_FILES, "dep-conf_", 286,	362, file);			// 76
	//generateFeaturesFor(2, 6, PATH_TO_FILES, "dep-conf_", 1685,	1858, file);		// 173
	//generateFeaturesFor(2, 6, PATH_TO_FILES, "dep-conf_", 1927,	2170, file);		// 243
	//generateFeaturesFor(2, SLOPEUP, PATH_TO_FILES, "dep-conf_", 2274,	2358, file);		// 84
	printf("training 2 slope up done\n"); // 576

	//7: slope down 
	//generateFeaturesFor(2, 7, PATH_TO_FILES, "dep-conf_", 576,	657, file);			// 81
	//generateFeaturesFor(2, 7, PATH_TO_FILES, "dep-conf_", 749,	982, file);			// 233
	//generateFeaturesFor(2, SLOPEDOWN, PATH_TO_FILES, "dep-conf_", 1069,	1176, file);		// 107
	//generateFeaturesFor(2, 7, PATH_TO_FILES, "dep-conf_", 2557,	2647, file);		// 90
	printf("training 2 slope down done\n"); // 511
}

void train3(FILE *file) {
	// TRAINING 3
	// 1: RUN
	generateFeaturesFor(3, 1, PATH_TO_FILES, "dep-conf_", 1734,	2060, file);		// 326
	printf("training 3 run done\n"); //326

	// 2: WALK 
	generateFeaturesFor(3, WALK, PATH_TO_FILES, "dep-conf_", 1,	159, file);			// 160
	generateFeaturesFor(3, 2, PATH_TO_FILES, "dep-conf_", 546,	582, file);			// 36
	generateFeaturesFor(3, 2, PATH_TO_FILES, "dep-conf_", 670, 818, file);			// 148
	generateFeaturesFor(3, 2, PATH_TO_FILES, "dep-conf_", 912,	1104, file);		// 192
	generateFeaturesFor(3, 2, PATH_TO_FILES, "dep-conf_", 1314,	1700, file);		// 386
	generateFeaturesFor(3, 2, PATH_TO_FILES, "dep-conf_", 2270,	2870, file);		// 600
	generateFeaturesFor(3, 2, PATH_TO_FILES, "dep-conf_", 3100,	3270, file);		// 170
	generateFeaturesFor(3, 2, PATH_TO_FILES, "dep-conf_", 3365,	3540, file);		// 175
	generateFeaturesFor(3, 2, PATH_TO_FILES, "dep-conf_", 3660,	3710, file);		// 50
	generateFeaturesFor(3, 2, PATH_TO_FILES, "dep-conf_", 3955,	4000, file);		// 45
	generateFeaturesFor(3, 2, PATH_TO_FILES, "dep-conf_", 4140,	4300, file);		// 160
	printf("training 3 walk done\n"); // 160+36+148+192+386+600+170+175+50+45+160=2122

	// 3: STAND
	generateFeaturesFor(3, STAND, PATH_TO_FILES, "dep-conf_", 2129, 2260, file);		// 131
	printf("training 3 stand done\n"); // 131

	// 4: stairs up
	generateFeaturesFor(3, STAIRSUP, PATH_TO_FILES, "dep-conf_", 2880,	3090, file);		// 210
	printf("training 3 stairs up done\n"); // 210

	// 5: stairs down
	generateFeaturesFor(3, STAIRSDOWN, PATH_TO_FILES, "dep-conf_", 1113,	1300, file);		// 187
	printf("training 3 stairs down done\n"); // 187

	// 6: slope up 
	//generateFeaturesFor(3, 6, PATH_TO_FILES, "dep-conf_",  175,	282, file);			// 107
	//generateFeaturesFor(3, SLOPEUP, PATH_TO_FILES, "dep-conf_",  335,	525, file);			// 190
	//generateFeaturesFor(3, 6, PATH_TO_FILES, "dep-conf_",  593,	658, file);			// 65
	//generateFeaturesFor(3, 6, PATH_TO_FILES, "dep-conf_",  3280, 3353, file);		// 73
	printf("training 3 slope up done\n"); // 107+190+65+73=435

	// 7: slope down 
	//generateFeaturesFor(3, SLOPEDOWN, PATH_TO_FILES, "dep-conf_", 829,	897, file);			// 68
	//generateFeaturesFor(3, SLOPEDOWN, PATH_TO_FILES, "dep-conf_", 3560,	3645, file);		// 85
	//generateFeaturesFor(3, 7, PATH_TO_FILES, "dep-conf_", 3725,	3940, file);		// 215
	//generateFeaturesFor(3, 7, PATH_TO_FILES, "dep-conf_", 4020,	4115, file);		// 95
	printf("training 3 slope down done\n");	// 463 
}

void train4(FILE *file) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/
	// TRAINING 4
	// run 1290	1480
	generateFeaturesFor(4, 1, PATH_TEST_4, "depth", 1290,	1480, file);
	// walk 
	/*
	1	210
	460	1270
	1740	2250
	*/
	generateFeaturesFor(4, 2, PATH_TEST_4, "depth", 1,	210, file);
	generateFeaturesFor(4, 2, PATH_TEST_4, "depth", 460,	1270, file);
	generateFeaturesFor(4, 2, PATH_TEST_4, "depth", 1740,	2250, file);

	// standing
	/*
	1540	1685
	3760	3830
	4345	4550
	*/
	generateFeaturesFor(4, 3, PATH_TEST_4, "depth", 1540,	1685, file);
	generateFeaturesFor(4, 3, PATH_TEST_4, "depth", 3760,	3830, file);
	generateFeaturesFor(4, 3, PATH_TEST_4, "depth", 4345,	4550, file);

	// stairs up 4
	// 2335	2530
	generateFeaturesFor(4, 4, PATH_TEST_4, "depth", 2335,	2530, file);

	// stairs down 5
	// 240	440
	generateFeaturesFor(4, 5, PATH_TEST_4, "depth", 240,	440, file);

	// slope up 6
	/*
	2800	2860
	4817	4955
	5081	5303
	*/
	//generateFeaturesFor(4, 6, PATH_TEST_4, "depth", 2800,	2860, file);
	//generateFeaturesFor(4, 6, PATH_TEST_4, "depth", 4817,	4955, file);
	//generateFeaturesFor(4, 6, PATH_TEST_4, "depth", 5081,	5303, file);

	// slope down 7
	/*
	3160	3275
	3530	3740
	3985	4141
	*/
	//generateFeaturesFor(4, 7, PATH_TEST_4, "depth", 3160,	3275, file);
	//generateFeaturesFor(4, 7, PATH_TEST_4, "depth", 3530,	3740, file);
	//generateFeaturesFor(4, 7, PATH_TEST_4, "depth", 3985,	4141, file);
}

void train5(FILE *file) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/
	// TRAINING 5
	// run 4715	5160
	generateFeaturesFor(5, 1, PATH_TEST_5, "depth", 4715,	5160, file);
	// walk  
	generateFeaturesFor(5, 2, PATH_TEST_5, "depth", 1,	160, file);
	generateFeaturesFor(5, 2, PATH_TEST_5, "depth", 390,	560, file);
	generateFeaturesFor(5, 2, PATH_TEST_5, "depth", 1650,	1820, file);
	generateFeaturesFor(5, 2, PATH_TEST_5, "depth", 2075,	2190, file);
	generateFeaturesFor(5, 2, PATH_TEST_5, "depth", 3242,	3430, file);
	generateFeaturesFor(5, 2, PATH_TEST_5, "depth", 3730,	3860, file);
	generateFeaturesFor(5, 2, PATH_TEST_5, "depth", 4125,	4370, file);
	generateFeaturesFor(5, 2, PATH_TEST_5, "depth", 5315,	5770, file); 
	
	// standing
	generateFeaturesFor(5, 3, PATH_TEST_5, "depth", 180,	195, file);
	generateFeaturesFor(5, 3, PATH_TEST_5, "depth", 785,	810, file);
	generateFeaturesFor(5, 3, PATH_TEST_5, "depth", 1840,	2025, file);
	generateFeaturesFor(5, 3, PATH_TEST_5, "depth", 4375,	4600, file);
	generateFeaturesFor(5, 3, PATH_TEST_5, "depth", 5220,	5295 , file);

	// stairs up 4
	generateFeaturesFor(5, 4, PATH_TEST_5, "depth", 5800,	5980, file);	 

	// stairs down 5
	// 3894	4100
	generateFeaturesFor(5, 5, PATH_TEST_5, "depth", 3894,	4100, file);

	// slope up 6
	//generateFeaturesFor(5, 6, PATH_TEST_5, "depth", 222,	267, file);
	//generateFeaturesFor(5, 6, PATH_TEST_5, "depth", 2210,	2315, file);
	//generateFeaturesFor(5, 6, PATH_TEST_5, "depth", 2590,	2820, file);
	//generateFeaturesFor(5, 6, PATH_TEST_5, "depth", 3060,	3155, file);
	
	 

	// slope down 7
	//generateFeaturesFor(5, 7, PATH_TEST_5, "depth", 640,	740, file);
	//generateFeaturesFor(5, 7, PATH_TEST_5, "depth", 950	,1200, file);
	//generateFeaturesFor(5, 7, PATH_TEST_5, "depth", 1500,	1630, file);
	//generateFeaturesFor(5, 7, PATH_TEST_5, "depth", 3505,	3570, file);
}

void train6(FILE *file) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/
	// TRAINING 6
	// run 
	generateFeaturesFor(6, 1, PATH_TEST_6, "depth", 3000,	3280, file); // 
	// walk  
	generateFeaturesFor(5, 2, PATH_TEST_6, "depth", 1600,	1790, file);
	generateFeaturesFor(5, 2, PATH_TEST_6, "depth", 3510,	4070, file);
	// standing
	 
	generateFeaturesFor(6, 3, PATH_TEST_6, "depth", 700,	760, file);
	generateFeaturesFor(6, 3, PATH_TEST_6, "depth", 3330,	3500, file);
	// stairs up 4
	generateFeaturesFor(6, 4, PATH_TEST_6, "depth", 4090,	4270, file);
	 
	// stairs down 5
	generateFeaturesFor(6, 5, PATH_TEST_6, "depth", 2255,	2440, file); 
	// slope up 6
	//generateFeaturesFor(5, 6, PATH_TEST_5, "depth", 222,	267, file);
	//generateFeaturesFor(5, 6, PATH_TEST_5, "depth", 2210,	2315, file);
	//generateFeaturesFor(5, 6, PATH_TEST_5, "depth", 2590,	2820, file);
	//generateFeaturesFor(5, 6, PATH_TEST_5, "depth", 3060,	3155, file);
	
	 

	// slope down 7
	//generateFeaturesFor(5, 7, PATH_TEST_5, "depth", 640,	740, file);
	//generateFeaturesFor(5, 7, PATH_TEST_5, "depth", 950	,1200, file);
	//generateFeaturesFor(5, 7, PATH_TEST_5, "depth", 1500,	1630, file);
	//generateFeaturesFor(5, 7, PATH_TEST_5, "depth", 3505,	3570, file);
}

void train7(FILE *file) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/
	// TRAINING 6
	// run  
	 
	generateFeaturesFor(7, 1, PATH_TEST_7, "depth", 9895,	10320, file); 
	// walk  
	generateFeaturesFor(7, 2, PATH_TEST_7, "depth", 5515,	6495, file);
	generateFeaturesFor(7, 2, PATH_TEST_7, "depth", 7735,	8658, file);
	// standing
	generateFeaturesFor(7, 3, PATH_TEST_7, "depth", 4680,	5385, file);
	generateFeaturesFor(7, 3, PATH_TEST_7, "depth", 6510,	7715, file);
	generateFeaturesFor(7, 3, PATH_TEST_7, "depth", 8660,	9764, file);
	  
	// stairs up 4
	generateFeaturesFor(7, 4, PATH_TEST_7, "depth", 2430,	2666, file);
	generateFeaturesFor(7, 4, PATH_TEST_7, "depth", 3197,	3440, file);
	generateFeaturesFor(7, 4, PATH_TEST_7, "depth", 3840,	4085, file); 
	 
	// stairs down 5 
	generateFeaturesFor(7, 5, PATH_TEST_7, "depth", 2045,	2285, file);
	generateFeaturesFor(7, 5, PATH_TEST_7, "depth", 2840,	3055, file);
	generateFeaturesFor(7, 5, PATH_TEST_7, "depth", 3565,	3775, file);
	generateFeaturesFor(7, 5, PATH_TEST_7, "depth", 4215,	4445, file);
}


void train8(FILE *file) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/
	// TRAINING 6
	// run  
	 
	generateFeaturesFor(8, 1, PATH_TEST_8, "depth", 5885,	7375, file); 
	// walk  
	generateFeaturesFor(8, 2, PATH_TEST_8, "depth", 3760,	5585, file); 
	// standing
	generateFeaturesFor(8, 3, PATH_TEST_8, "depth", 3440,	3750, file); 
	  
	// stairs up 4
	generateFeaturesFor(8, 4, PATH_TEST_8, "depth", 1425,	1650, file); 
	 
	// stairs down 5 
	generateFeaturesFor(8, 5, PATH_TEST_8, "depth", 950,	1160, file); 
}

void train9(FILE *file) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/
	// TRAINING 6
	// run  
	 
	generateFeaturesFor(9, 1, PATH_TEST_9, "depth", 3460,	3700, file); 
	// walk  
	generateFeaturesFor(9, 2, PATH_TEST_9, "depth", 2190,	2685, file); 
	generateFeaturesFor(9, 2, PATH_TEST_9, "depth", 4150,	4740, file); 
	// standing
	generateFeaturesFor(9, 3, PATH_TEST_9, "depth", 2695,	3400, file); 
	generateFeaturesFor(9, 3, PATH_TEST_9, "depth", 3720,	4140, file); 
	  
	// stairs up 4
	generateFeaturesFor(9, 4, PATH_TEST_9, "depth", 4755,	4985, file); 
	 
	// stairs down 5 
	generateFeaturesFor(9, 5, PATH_TEST_9, "depth", 1890,	2145, file); 
}
 
void train9All(FILE *file) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/ 
	generateFeaturesFor(9, 3, PATH_TEST_9, "depth", 6,	1570, file); // stand
	printf("1st done\n");
	generateFeaturesFor(9, 2, PATH_TEST_9, "depth", 1571,	1889, file);  // walk 
	printf("2st done\n");
	generateFeaturesFor(9, 5, PATH_TEST_9, "depth", 1890,	2158, file); // stairs down 
	printf("3st done\n");
	generateFeaturesFor(9, 2, PATH_TEST_9, "depth", 2159,	2696, file);  // walk
	printf("4st done\n");
	generateFeaturesFor(9, 3, PATH_TEST_9, "depth", 2697,	3400, file); // stand
	printf("5th done\n");
	generateFeaturesFor(9, 2, PATH_TEST_9, "depth", 3401,	3428, file);  // walk
	printf("6 done\n");
	generateFeaturesFor(9, 1, PATH_TEST_9, "depth", 3429,	3715, file); // run
	printf("7 done\n");
	generateFeaturesFor(9, 3, PATH_TEST_9, "depth", 3716,	4130, file); // stand
	printf("8 done\n");
	generateFeaturesFor(9, 2, PATH_TEST_9, "depth", 4131,	4740, file); // walk 
	printf("9 done\n");
	generateFeaturesFor(9, 4, PATH_TEST_9, "depth", 4741,	4990, file); // satirs up
	printf("10 done\n");
	generateFeaturesFor(9, 2, PATH_TEST_9, "depth", 4991,	5060, file); // walk 
	printf("11 done\n");
}

void trainSet1(FILE *file, char *path) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/   
	printf("TRAIN SET 1\n");
	generateFeaturesFor(1, 3, path, "depth", 6,	686, file);  // stand 
	printf("1st done\n");
	generateFeaturesFor(1, 2, path, "depth", 687, 921, file); 
	printf("1st done\n");
	generateFeaturesFor(1, 5, path, "depth", 922, 1126, file); 
	printf("1st done\n");
	generateFeaturesFor(1, 2, path, "depth", 1127, 1621, file); 
	printf("1st done\n");
	generateFeaturesFor(1, 1, path, "depth", 1622, 1885, file); 
	printf("1st done\n");
	generateFeaturesFor(1, 3, path, "depth", 1886, 2046, file); 
	printf("1st done\n");
	generateFeaturesFor(1, 2, path, "depth", 2047, 2792, file); 
	printf("1st done\n");
	generateFeaturesFor(1, 4, path, "depth", 2793, 2986, file); 
	printf("1st done\n");
	generateFeaturesFor(1, 2, path, "depth", 2987, 3200, file); 
	printf("1st done\n");
}
 

void trainSet2(FILE *file, char *path) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/   
	printf("TRAIN SET 2\n");
	generateFeaturesFor(2, WALK, path, "depth", 3562, 3695, file);  
	printf("done \n");
	generateFeaturesFor(2, STAIRSDOWN, path, "depth", 3696, 3922, file);  
	printf("done \n");
	generateFeaturesFor(2, WALK, path, "depth", 3923, 4065, file);  
	printf("done \n");
	generateFeaturesFor(2, RUN, path, "depth", 4066, 4413, file);  
	printf("done \n");
	generateFeaturesFor(2, WALK, path, "depth", 4414, 5489, file);  
	printf("done \n");
	generateFeaturesFor(2, STAIRSUP, path, "depth", 5490, 5712, file);  
	printf("done \n");
	generateFeaturesFor(2, WALK, path, "depth", 5713, 5987, file);  
	  
}
void trainSet3(FILE *file, char *path) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/   
	printf("TRAIN SET 3\n");
	generateFeaturesFor(3, WALK, path, "depth", 2256, 2368, file);  
	printf("1st done\n");
	generateFeaturesFor(3, STAIRSDOWN, path, "depth", 2369, 2570, file);  
	printf("1st done\n");
	generateFeaturesFor(3, WALK, path, "depth", 2571, 2989, file);  
	printf("1st done\n");
	generateFeaturesFor(3, RUN, path, "depth", 2990, 3330, file);  
	printf("1st done\n");
	generateFeaturesFor(3, WALK, path, "depth", 3331, 3398, file);  
	printf("1st done\n");
	generateFeaturesFor(3, STAND, path, "depth", 3399, 3504, file);  
	printf("1st done\n");
	generateFeaturesFor(3, WALK, path, "depth", 3505, 4135, file);  
	printf("1st done\n");
	generateFeaturesFor(3, STAIRSUP, path, "depth", 4136, 4366, file);  
	printf("1st done\n");
	generateFeaturesFor(3, WALK, path, "depth", 4367, 4520, file);  
	printf("1st done\n");
	 
}
void trainSet4(FILE *file, char *path) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/   
	printf("TRAIN SET 4\n");
	generateFeaturesFor(4, STAND, path, "depth", 550, 686, file);  
	printf("1st done\n");
	generateFeaturesFor(4, WALK, path, "depth", 687, 925, file);  
	printf("1st done\n");
	generateFeaturesFor(4, STAIRSDOWN, path, "depth", 926, 1130, file);  
	printf("1st done\n");
	generateFeaturesFor(4, WALK, path, "depth", 1131, 1856, file);  
	printf("1st done\n");
	
	generateFeaturesFor(4, WALK, path, "depth", 1869, 2008, file);  
	printf("1st done\n");
	generateFeaturesFor(4, RUN, path, "depth", 2009, 2187, file);  
	printf("1st done\n");
	generateFeaturesFor(4, WALK, path, "depth", 2188, 2229, file);  
	printf("1st done\n");
	generateFeaturesFor(4, STAND, path, "depth", 2230, 2382, file);  
	printf("1st done\n");
	generateFeaturesFor(4, WALK, path, "depth", 2383, 3003, file);  
	printf("1st done\n");
	generateFeaturesFor(4, STAIRSUP, path, "depth", 3004, 3221, file);  
	printf("1st done\n");
}

//#define TRAIN_SET_INCLUDED_PART1

void trainSet5(FILE *file, char *path) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/   
	printf("TRAIN SET 5\n");
	generateFeaturesFor(5, WALK, path, "depth", 4631,	4834, file);  // walk 
	printf("1st done\n");
	generateFeaturesFor(5, STAIRSDOWN, path, "depth", 4835,	5061, file);  // walk 
	printf("1st done\n");
	generateFeaturesFor(5, WALK, path, "depth", 5062,	5333, file);  // walk 
	printf("1st done\n");
	generateFeaturesFor(5, STAND, path, "depth", 5334,	5565, file);  // walk 
	printf("1st done\n");
	generateFeaturesFor(5, WALK, path, "depth", 5566,	5657, file);  // walk 
	printf("1st done\n");
	generateFeaturesFor(5, RUN, path, "depth", 5658,	6115, file);  // walk 
	printf("1st done\n");
	generateFeaturesFor(5, STAND, path, "depth", 6116,	6253, file);  // walk 
	printf("1st done\n");
	generateFeaturesFor(5, WALK, path, "depth", 6254,	6733, file);  // walk 
	printf("1st done\n");
	generateFeaturesFor(5, STAIRSUP, path, "depth", 6734,	6940, file);  // walk 
	printf("1st done\n");
}

void trainSet6(FILE *file, char *path) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/   
	printf("TRAIN SET 6\n");
	generateFeaturesFor(6, STAND, path, "depth", 3600,	3660, file);  // walk  
	printf("1st done\n"); 
	generateFeaturesFor(6, WALK, path, "depth", 3661,	3817, file);  // walk  
	printf("1st done\n"); 
	generateFeaturesFor(6, STAIRSDOWN, path, "depth", 3818,	4021, file);  // walk  
	printf("1st done\n"); 
	generateFeaturesFor(6, WALK, path, "depth", 4022,	4553, file);  // walk  
	printf("1st done\n"); 
	generateFeaturesFor(6, RUN, path, "depth", 4554,	4845, file);  // walk  
	printf("1st done\n"); 
	generateFeaturesFor(6, WALK, path, "depth", 4846,	4892, file);  // walk  
	printf("1st done\n"); 
	generateFeaturesFor(6, STAND, path, "depth", 4893,	5070, file);  // walk  
	printf("1st done\n"); 
	generateFeaturesFor(6, WALK, path, "depth", 5071,	5645, file);  // walk  
	printf("1st done\n"); 
	generateFeaturesFor(6, STAIRSUP, path, "depth", 5646,	5838, file);  // walk  
	printf("1st done\n"); 
}

void trainSet7(FILE *file, char *path) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/   
	printf("TRAIN SET 7\n");
	generateFeaturesFor(9, 2, path, "depth", 2708,	3022, file);  // walk  
	printf("1st done\n");
	generateFeaturesFor(9, 5, path, "depth", 3023,	3202, file); // stairs down 
	printf("2st done\n");
	generateFeaturesFor(9, 2, path, "depth", 3203,	3362, file);  // walk
	printf("3st done\n");
	generateFeaturesFor(9, 3, path, "depth", 3363,	4446, file); // stand
	printf("3th done\n");
	generateFeaturesFor(9, 2, path, "depth", 4447,	5169, file);  // walk
	printf("5 done\n");
	generateFeaturesFor(9, 3, path, "depth", 5170,	7484, file); // stand 
	
	printf("6 done\n");
	generateFeaturesFor(9, 1, path, "depth", 7485,	7828, file); // run
	printf("7 done\n");
	generateFeaturesFor(9, 3, path, "depth", 7829,	7989, file); // stand 
	printf("9 done\n");
	generateFeaturesFor(9, 4, path, "depth", 7990,	8227, file); // satirs up
	printf("10 done\n"); 
}


void trainSet8(FILE *file, char *path) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/   
	printf("TRAIN SET 8\n");
	generateFeaturesFor(9, 2, path, "depth", 1363,	1705, file);  // walk  
	printf("1st done\n");
	generateFeaturesFor(9, 5, path, "depth", 1706,	1906, file); // stairs down 
	printf("2st done\n");
	generateFeaturesFor(9, 2, path, "depth", 1907,	2790, file);  // walk
	printf("3st done\n");
	generateFeaturesFor(9, 3, path, "depth", 2791,	3557, file); // stand  
	printf("6 done\n");
	generateFeaturesFor(9, 1, path, "depth", 3558,	3931, file); // run
	printf("7 done\n");
	generateFeaturesFor(9, 3, path, "depth", 3932,	4010, file); // stand 
	printf("9 done\n");
	generateFeaturesFor(9, 4, path, "depth", 4011,	4251, file); // satirs up 
	printf("11 done\n");
}


void trainSet9(FILE *file, char *path) {
	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/   
	printf("TRAIN SET 9\n");
	generateFeaturesFor(9, 3, path, "depth", 6,	1580, file);  
	printf("1st done\n");
	generateFeaturesFor(9, 2, path, "depth", 1581,	1888, file); 
	printf("2st done\n");
	generateFeaturesFor(9, 5, path, "depth", 1889,	2125, file);  
	printf("3st done\n");
	generateFeaturesFor(9, 2, path, "depth", 2126,	2700, file); 
	printf("4 done\n");
	generateFeaturesFor(9, 3, path, "depth", 2701,	3410, file); 
	printf("5 done\n");
	generateFeaturesFor(9, 1, path, "depth", 3411,	3699, file); 
	printf("6 done\n");
	generateFeaturesFor(9, 3, path, "depth", 3700,	4137, file); 
	printf("7 done\n");
	generateFeaturesFor(9, 2, path, "depth", 4138,	4740, file); 
	printf("8 done\n");
	generateFeaturesFor(9, 4, path, "depth", 4741,	4985, file); 
	printf("9 done\n"); 
}
#define SIX_TRAIN_ONLY
void generateFeatures() {

	/* 
	1: running
	2: walking
	3: standing
	4: stairs up
	5: stairs down
	6: slope up
	7: slope down
	*/
	
#ifndef SIX_TRAIN_ONLY
	FILE *features1 = fopen("features1.txt", "w");
	FILE *features2 = fopen("features2.txt", "w");
	FILE *features3 = fopen("features3.txt", "w");
	FILE *features4 = fopen("features4.txt", "w");
	FILE *features5 = fopen("features5.txt", "w");
	FILE *features6 = fopen("features6.txt", "w");
	FILE *features7 = fopen("features7.txt", "w");
	FILE *features8 = fopen("features8.txt", "w");
	FILE *features9 = fopen("features9.txt", "w");
	FILE *features9All = fopen("features9All.txt", "w");
#endif

#ifdef TRAIN_SET_INCLUDED_PART1
	FILE *featuresSet1 = fopen("featuresSet1.txt", "w");
	FILE *featuresSet2 = fopen("featuresSet2.txt", "w");
	FILE *featuresSet3 = fopen("featuresSet3.txt", "w");
	FILE *featuresSet4 = fopen("featuresSet4.txt", "w");
	FILE *featuresSet5 = fopen("featuresSet5.txt", "w");
	FILE *featuresSet6 = fopen("featuresSet6.txt", "w");
	FILE *featuresSet7 = fopen("featuresSet7.txt", "w");
	FILE *featuresSet8 = fopen("featuresSet8.txt", "w");	
#endif
	FILE *featuresSet6 = fopen("featuresSet6.txt", "w");
	
#ifndef SIX_TRAIN_ONLY
	printf("dataset 1 starting\n");
	train1(features1);
	printf("dataset 2 starting\n");
	train2(features2);
	printf("dataset 3 starting\n");
	train3(features3);
	printf("dataset 4 starting\n");
	train4(features4);
	printf("dataset 5 starting\n");
	train5(features5);
	printf("dataset 6 starting\n");
	train6(features6);
	train7(features7);
	train8(features8);
	train9(features9);
	train9All(features9All);
#endif
#ifdef TRAIN_SET_INCLUDED_PART1
	trainSet1(featuresSet1, PATH_SET_1);
	trainSet2(featuresSet2, PATH_SET_2);
	trainSet3(featuresSet3, PATH_SET_3);
	trainSet4(featuresSet4, PATH_SET_4);
	trainSet5(featuresSet5, PATH_SET_5);
	trainSet6(featuresSet6, PATH_SET_6);
	trainSet7(featuresSet7, PATH_SET_7);
	trainSet8(featuresSet8, PATH_SET_8);
#endif
	trainSet6(featuresSet6, PATH_SET_6);

#ifndef SIX_TRAIN_ONLY
	fclose(features1);
	fclose(features2);
	fclose(features3);
	fclose(features4);
	fclose(features5);
	fclose(features6);
	fclose(features7);
	fclose(features8);
	fclose(features9);
	fclose(features9All); 
#endif
#ifdef TRAIN_SET_INCLUDED_PART1
	fclose(featuresSet1);
	fclose(featuresSet2);
	fclose(featuresSet3);
	fclose(featuresSet4);
	fclose(featuresSet5); 
	fclose(featuresSet6); 
	fclose(featuresSet7); 
	fclose(featuresSet8); 	
#endif
	fclose(featuresSet6); 
}