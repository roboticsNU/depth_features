#include "features/feature_extractor.h"
#include "camera.h"
#include "serial.h"
#include <iostream> 
#include "EasiiSDK/Iisu.h" 

#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui_c.h" 
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/videoio.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2\core\core.hpp>
#include<time.h>   
#include <stdio.h>
#include <math.h>
#include "svmlib\svm.h"
 
#include <Windows.h>

using namespace SK::Easii;  

bool volatile finished = false;
bool volatile onlySerial = false;

DWORD WINAPI cameraStart(CONST LPVOID lpParam) {
	camera();
	ExitThread(0);
}

DWORD WINAPI serialStart(CONST LPVOID lpParam) {
	serial();
	ExitThread(0);
}

DWORD WINAPI communicationStart(CONST LPVOID lpParam) {
	serial();
	ExitThread(0);
}

//#define TRAINING 
//#define MODEL_CREATING
int main(int argc, char **argv) { 
#ifdef TRAINING
	generateFeatures();
	return 0;
#endif
#ifdef MODEL_CREATING
	char *args[5] = {"", "-v", "10", "C:\\robotics\\training\\features.txt", "C:\\robotics\\training\\model"};
	return main_train(4, args);
#endif
	HANDLE camThread;
	HANDLE serialThread;
	HANDLE commThread;
	HANDLE threads[2];
	camThread = CreateThread(NULL, 0, &cameraStart, NULL, 0, NULL);
	if(NULL == camThread) {
		printf("Failed to create thread: CAMERA.\r\n");
    }
	
	serialThread = CreateThread(NULL, 0, &serialStart, NULL, 0, NULL);
	if(NULL == camThread) {
		printf("Failed to create thread: SERIAL.\r\n");
    }

	commThread = CreateThread(NULL, 0, &communicationStart, NULL, 0, NULL);
	if(NULL == camThread) {
		printf("Failed to create thread: COMMUNICATION.\r\n");
    }
	

	threads[0] = camThread;
	threads[1] = serialThread;
	 
	char value = 0;
	scanf("%c", &value);
	finished = true;
	
	printf("MAIN FINISH START\n");
	
	WaitForMultipleObjects(2, threads, TRUE, INFINITE);
	for(int i = 0; i < 2; i++) {
		CloseHandle(threads[i]);
	}
	printf("MAIN FINISHED\n");
}
