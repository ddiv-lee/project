#include "throw_algo.h"

#include <math.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "opencv2/opencv.hpp"
//#include "utils/ein_log.h"

using namespace cv;
using namespace std;

namespace EinBox {
HaltThrowAlgo::HaltThrowAlgo() { first_frame_flag_ = true; }
HaltThrowAlgo::~HaltThrowAlgo() {}

int HaltThrowAlgo::handleFrame(cv::Mat &frame, std::vector<cv::Rect> &res) {
  static int cont = 0;
  cout << "~~~" << endl;

  if (frame.empty()) {
    std::cout << "Input frame is empty!" << std::endl;
    return -1;
  }
  Mat result,gray,FGModel;
  Res_Motion_Detection motion_res;
  cvtColor(frame, gray, COLOR_RGB2GRAY);
  vector<Point2d> prebboxes;
  if (first_frame_flag_) {
    vibe.init(gray);
    vibe.ProcessFirstFrame(gray);
   
    std::cout << "Training halt throw complete!" << std::endl;
    first_frame_flag_ = false;
  } else {
    vibe.Run(gray);
    FGModel = vibe.getFGModel();
    motion_res = motion_detection(FGModel, frame);
  }

  throw_object_res =
      object_detection(frame, motion_res.center_location,
                       throw_object_res.prebboxes, throw_object_res.L);

  // cout << "di " << cont << " zhen :" << endl;
  res = throw_object_res.res_box;
  for (size_t i = 0; i < res.size(); i++) {
    std::cout << "Capture throw thing, x: " << res[i].x << ", y: " << res[i].y
              << ", w: " << res[i].x + res[i].width
              << ", h : " << res[i].y + res[i].height << std::endl;
	/*
	rectangle(frame, Point(res[i].x, res[i].y),
		        Point(res[i].x + 100, res[i].y + 100),
		         Scalar(255, 0, 0), 2, 8);
	*/
	circle(frame, Point(res[i].x, res[i].y), 10, (0, 0, 255), 4);
  }
  imshow("org", frame);
  waitKey(1);
  // // cout << "map length:" << throw_object_res.L.size()<<"\n";
  // // imshow("throw_object_result", throw_object_res.result);
  // for (map<double, int>::iterator it = throw_object_res.L.begin();
  //      it != throw_object_res.L.end(); ++it) {
  //   // cout << "test";
  //   // cout <<"throw_object_res.L=" <<it->first << " => " << it->second <<
  //   '\n';
  // }

  if (motion_res.center_location.size() != 0) {
    throw_object_res.prebboxes = motion_res.center_location;
  }
  previousframe = frame.clone();
  cont++;

  return 0;
}

Res_Motion_Detection HaltThrowAlgo::motion_detection(Mat FGModel,
                                                     Mat tempframe) {
  res_motion_detection res;
  //res.result = tempframe.clone();
  
  morphologyEx(FGModel, FGModel, MORPH_OPEN, Mat());
 

  // findContours
  vector<vector<Point> > contours;
  findContours(FGModel, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  vector<Rect> boundRect(contours.size());
  Point2d temp;
  for (size_t i = 0; i < contours.size(); i++) {
    if (contourArea(contours[i]) < 4 ||
        contourArea(contours[i]) > 16000) {  // filter
      continue;
    }
    Point2d center_xy;
    boundRect[i] = boundingRect(contours[i]);
    // cout << "bbox=" <<boundRect[i].tl()<<"\n";
    // cout << "bbox_r=" << boundRect[i].br() << "\n";
    center_xy.x = (boundRect[i].br().x + boundRect[i].tl().x) / 2;
    center_xy.y = (boundRect[i].br().y + boundRect[i].tl().y) /2;  //count center
                    
    if (i == 0) {
      res.center_location.push_back(center_xy);
      //Scalar scalar(0, 0, 255);
    }
    if (i > 0) {
      int distance = powf(temp.x - center_xy.x, 2) + powf(temp.y - center_xy.y, 2);
      if (distance > 8000) {
        //Scalar scalar(0, 0, 255);
        //circle(res.result, center_xy, 10, scalar, 4);
        res.center_location.push_back(center_xy);
      }
    }
    temp = center_xy;
  }
  return res;  //return result
}

Res_Object_Detection HaltThrowAlgo::object_detection(Mat frame,
                                                     vector<Point2d> bboxes,
                                                     vector<Point2d> prebboxes,
                                                     map<double, int> L) {
  Res_Object_Detection res;
  //res.result = frame.clone();
  if (bboxes.size() <= 15) {
	  for (size_t i = 0; i < bboxes.size(); i++) {
		  for (size_t j = 0; j < prebboxes.size(); j++) {
			  int distance = powf(bboxes[i].x - prebboxes[j].x, 2) +
				  powf(bboxes[i].y - prebboxes[j].y, 2);
			  if (distance < 1000) {
				  int delth = bboxes[i].y - prebboxes[j].y;
				  if (delth >= 5 && delth < 50) {
					  /*
					  cout << "bboxes=" << bboxes[i]<<"\n";
					  cout << "prebboxes=" << prebboxes[i]<<"\n";
					  cout << "delth" << delth<<"\n";*/
					  double key = prebboxes[j].x * 1000 + prebboxes[j].y;
					  double current_key = bboxes[i].x * 1000 + bboxes[i].y;
					  if (L.find(key) != L.end()) {
						  int predelth = L[key];
						  res.L[current_key] = predelth + delth;
						  if (res.L[current_key] > 10) {
							  // circle(res.result, bboxes[i], 10, (0, 0, 255), 4);
							  //rectangle(res.result, Point(bboxes[i].x - 10, bboxes[i].y - 10),
							  //          Point(bboxes[i].x + 10, bboxes[i].y + 10),
							  //          Scalar(0, 0, 255), 2, 8);
							  Rect rect(bboxes[i].x - 10, bboxes[i].y - 10, 20, 20);
							  res.res_box.push_back(rect);
						  }
					  }
					  else {


						  res.L[current_key] = delth;
					  }
				  }
			  }
		  }
	  }
  }
  return res;
}
}  // namespace EinBox