#ifndef __THROW_ALGO_H__
#define __THROW_ALGO_H__

#include <math.h>

#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/opencv.hpp"
#include "Vibe.h"

using namespace cv;
using namespace std;

namespace EinBox {
typedef struct res_motion_detection {
  //Mat result;
  vector<Point2d> center_location;
} Res_Motion_Detection;

typedef struct res_object_detection {
  //Mat result;
  vector<Point2d> prebboxes;
  map<double, int> L;
  vector<Rect> res_box;
} Res_Object_Detection;

class HaltThrowAlgo {
 public:
  HaltThrowAlgo();
  virtual ~HaltThrowAlgo();
  int handleFrame(cv::Mat &frame, std::vector<cv::Rect> &res);

 private:
  //void processFirstFrame(const Mat _image);

  Res_Motion_Detection motion_detection(Mat FGModel, Mat tempframe);
  Res_Object_Detection object_detection(Mat frame, vector<Point2d> bboxes,
                                        vector<Point2d> prebboxes,
                                        map<double, int> L);

 private:
  bool first_frame_flag_;
  ViBe vibe;
  Res_Object_Detection throw_object_res;
  Mat previousframe;
};

}  // namespace EinBox
#endif
