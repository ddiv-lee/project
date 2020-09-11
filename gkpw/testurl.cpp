#include"opencv2/opencv.hpp"  
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <ctime>
#include <iostream>
#include"throw_algo.h"
#include <thread>  
#include <mutex> 
#include <chrono>
#include <stack>
#include <string>
#include <time.h>
#include <stack>

//»¥³â·ÃÎÊ  
#include <mutex>  
using namespace std;
using namespace cv;

using namespace EinBox;
cv::VideoCapture capture;
std::mutex mtx;
std::string url = "rtsp://admin:123456@192.168.2.211:554/ch01.264?dev=1";
/**
* @brief ¶ÁÍ¼
*
* @return Mat
*/
void captureThread(stack<cv::Mat> &frame_q)
{
	cv::Mat frame;

	//´ò¿ªÍ¼Ïñ  
	capture.open(url);


	while (1)
	{
		//cout << "111111" << std::endl;
		//¼ÓËø  
		mtx.lock();
		if (frame_q.size()>50) {
			while (!frame_q.empty()) {
				frame_q.pop();

			}
		}
		capture >> frame;
		//cout << "thread1" << std::endl;
		frame_q.push(frame);
		//cout << "push" << std::endl;
		//½âËø  
		mtx.unlock();
		std::this_thread::sleep_for(std::chrono::milliseconds(40));
		//cout << "end" << std::endl;
	}
}

int main()
{
	HaltThrowAlgo throwalgo;
	vector<Rect> res_box;
	stack<cv::Mat> frame_q;
	thread t1(captureThread, ref(frame_q));
	t1.detach();
	cv::Mat frame;
	clock_t start, end;
	int i = 0;
	for (;;) {
		//cout << "in for" << std::endl;
		mtx.lock();
		if (!frame_q.empty()) {
			//cout << "copy" << std::endl;
			frame_q.top().copyTo(frame);
			frame_q.pop();
		}
		mtx.unlock();
		if (frame.rows>0) {
			start = clock();
			throwalgo.handleFrame(frame, res_box);
			end = clock();
			double endtime = (double)(end - start) / CLOCKS_PER_SEC;
			if (endtime > 0.15) {
				cout << i << "cost:" << endtime << endl;
			}
			
			//cout << "imshow" << std::endl;
			//imshow("Live", frame);
			//cv::waitKey(1);
			//cout << "3333333" << std::endl;
		}
		i++;
	}
}

