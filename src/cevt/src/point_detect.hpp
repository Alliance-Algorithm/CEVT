#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/ForwardDeclarations.h>
#include <eigen3/Eigen/src/Geometry/Quaternion.h>

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <queue>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <string>
#include <vector>

#include "../include/pnp/pnp_solver.hpp"

namespace EngineerVisual {

class PointDetect {
public:
  PointDetect() : pnpsolver() {

    //初始化相机参数

    pnpsolver.SetCameraMatrix(1.722231837421459e+03, 1.724876404292754e+03,
                              7.013056440882832e+02, 5.645821718351237e+02);
    //设置畸变参数
    pnpsolver.SetDistortionCoefficients(-0.064232403853946, -0.087667493884102,
                                        0, 0, 0.792381808294582);
    for (int i = 0; i < 10; i++) {
      filter.push({});
    }
  };
  void Calculate(cv::Mat &image) {
    original = cv::Mat(image);
    cv::Mat roi;
    cv::Point2f rCenter(0, 0);
    cv::RotatedRect roiRect;
    float rRadius = 0;
    Preprocessing(image);
    if (!GetR(image, rCenter, rRadius)) {
#ifdef SHOW_IMG
      cv::imshow("roi", image);
      cv::waitKey(10);
#endif
      return;
    }
    if (!GetFlyswatter(image, roi, rCenter, roiRect)) {
#ifdef SHOW_IMG
      cv::imshow("roi", image);
      cv::waitKey(10);
#endif
      return;
    }

    Preprocessing(roi);
    Processing(roi, rCenter, roiRect);
  }
  Eigen::Quaternionf &Rotate() { return rotate; }
  Eigen::Vector3f &Position() { return position; }

private:
  void Preprocessing(cv::Mat &image) {
    // 颜色过滤
    std::vector<cv::Mat> rgb;
    cv::split(image, rgb);
    // image = rgb.at(0) - rgb.at(2);
    // //转成灰度图片
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    // #二值化
    //     ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY);
    cv::threshold(image, image, 100, 255, cv::THRESH_BINARY);
  };

  //获取R标
  bool GetR(cv::Mat &image, cv::Point2f &center, float &radius) {
    std::vector<std::vector<cv::Point>> outlines{};
    std::vector<cv::Vec4i> hierarchies{};
    int min = 10000, index = -1;

    cv::findContours(image, outlines, hierarchies, cv::RETR_TREE,
                     cv::CHAIN_APPROX_NONE);

    for (int i = 0, outlines_size = outlines.size(); i < outlines_size; i++) {

#ifdef SHOW_IMG
      cv::drawContours(image, outlines, i, cv::Scalar(100), 4);
#endif
      double area = cv::contourArea(outlines[i]);

      if (area < 10 || area > 10000)
        continue;
      /*找到没有父轮廓的轮廓*/
      if (hierarchies[i][3] >= 0 && hierarchies[i][3] < outlines_size)
        continue;
      /*找有子轮廓的*/
      if (hierarchies[i][2] < 0 || hierarchies[i][2] >= outlines_size)
        continue;
      if (min >= area) {
        min = area;
        index = i;
      }
    }
    if (index == -1) {
      // cv::imshow("NO R Pos", image);
      // cv::waitKey(10);
      return false;
    }
    cv::minEnclosingCircle(cv::Mat(outlines[index]), center, radius);
    // cv::circle(image, center, radius, cv::Scalar(200), -1, 8, 0);
#ifdef SHOW_IMG
    cv::circle(original, center, radius, cv::Scalar(200), -1, 8, 0);
#endif
    // cv::imshow("R Pos", image);
    // cv::waitKey(10);
    return true;
  }

  //获取苍蝇拍子
  bool GetFlyswatter(cv::Mat &image, cv::Mat &roi, cv::Point2f &rCenter,
                     cv::RotatedRect &roiRect) {
    cv::Mat element = getStructuringElement(0, cv::Size(5, 5));
    cv::Mat dilateMat = cv::Mat(image);

    // cv::morphologyEx(image, dilateMat, cv::MORPH_CLOSE, element);
    // cv::dilate(image, dilateMat, element, cv::Point(-1, -1), 2);

    std::vector<std::vector<cv::Point>> outlines{};
    std::vector<cv::Vec4i> hierarchies{};
    int max = 1000, index = -1;

    findContours(dilateMat, outlines, hierarchies, cv::RETR_TREE,
                 cv::CHAIN_APPROX_NONE);

    for (int i = 0, outlines_size = outlines.size(); i < outlines_size; i++) {
      // cv::drawContours(original, outlines, i, cv::Scalar(100), 4);
      double area = cv::contourArea(outlines[i]);
      if (area < 10 || area > 10000)
        continue;
      /*找到没有父轮廓的轮廓*/
      if (hierarchies[i][3] >= 0 && hierarchies[i][3] < outlines_size)
        continue;
      /*找没子轮廓的*/
      if (hierarchies[i][2] >= 0 && hierarchies[i][2] < outlines_size) {
        double area2 = cv::contourArea(outlines[hierarchies[i][2]]);
        if (outlines[hierarchies[i][2]].size() > 2 && area2 < 1000.0)
          continue;
      }
      if (max <= area) {
        max = area;
        index = i;
      }
    }
    // cv::drawContours(original, outlines, index, cv::Scalar(100, 100, 255),
    // 4);

    if (index == -1)
      return false;
    roiRect = cv::minAreaRect(outlines[index]);
    cv::Point2f box[4];
    roiRect.points(box);
#ifdef SHOW_IMG
    cv::circle(image, box[0], 10, cv::Scalar(200), -1, 8, 0);
#endif

    int boxIndex[2] = {-1, -1};
    float boxMaxlength[2]{0, 0};
    for (int i = 0; i < 4; i++) {
      float l = pow(rCenter.x - box[i].x, 2) + pow(rCenter.y - box[i].y, 2);
      if (l < boxMaxlength[0]) {
        if (l < boxMaxlength[1])
          continue;
        boxMaxlength[1] = l;
        boxIndex[1] = i;
        continue;
      }
      boxMaxlength[1] = boxMaxlength[0];
      boxIndex[1] = boxIndex[0];
      boxMaxlength[0] = l;
      boxIndex[0] = i;
    }

    if (boxIndex[0] == -1 || boxIndex[0] == -1)
      return false;

    cv::Point2f forward =
        ((box[boxIndex[0]] + box[boxIndex[1]]) / 2 - rCenter) * 1 / 2;

    box[boxIndex[0]] += forward;
    box[boxIndex[1]] += forward;

    const cv::Point *pp[] = {new cv::Point[]{box[0], box[1], box[2], box[3]}};
    int n[] = {4};

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::fillPoly(mask, pp, n, 1, cv::Scalar(255));

    cv::copyTo(original, roi, mask);

    // cv::imshow("roi", roi);

    return true;
  }

  void Processing(cv::Mat &image, cv::Point2f rCenter,
                  cv::RotatedRect roiRect) {

    std::vector<std::vector<cv::Point>> cors;
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    // // 计算 x 方向的梯度
    // Sobel(image, grad_x, CV_16S, 1, 0, 3);
    // convertScaleAbs(grad_x, abs_grad_x);

    // // 计算 y 方向的梯度
    // Sobel(image, grad_y, CV_16S, 0, 1, 3);
    // convertScaleAbs(grad_y, abs_grad_y);

    // // 合并梯度（近似）
    // cv::Mat grad;
    // addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, image);

    cv::imshow("roi", image);
    cv::waitKey(10);

    cv::findContours(image, cors, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    int sign;

    std::vector<cv::Point2f> convex;
    for (auto cor : cors) {

      if (cv::contourArea(cor) < 800)
        continue;

      double epsilon = cv::arcLength(cor, true) * 0.03;
      cv::approxPolyDP(cor, cor, epsilon, true);

      for (int cor_i_size = (int)cor.size(), j = cor_i_size,
               dcor_i_size = 2 * cor_i_size;
           j < dcor_i_size; j++) {
        if ((cor[(j - 1) % cor_i_size] - cor[j % cor_i_size])
                .cross(cor[(j + 1) % cor_i_size] - cor[j % cor_i_size]) > 0) {
          convex.push_back(cor[j % cor_i_size]);
        }
      }
    }

    std::vector<cv::Point2f> convexFilted;
    auto roiCenter = roiRect.center;
    roiCenter = roiCenter + (rCenter - roiCenter) * 0.3;
    for (auto point : convex) {
      if ((roiCenter - point).dot(rCenter - point) > 0) {
        convexFilted.push_back(point);
#ifdef SHOW_IMG
        cv::circle(original, point, 10, cv::Scalar(100, 100, 100));
#endif
      }
    }
    if (convexFilted.size() <= 0)
      return;

    cv::RotatedRect head = cv::minAreaRect(convexFilted);

    cv::Point2f box[4];
    cv::Point2f sortedBox[4]; // 0 as LT,1 as RT,2 as RB, 3 as LB
    head.points(box);

    for (auto h : box) {
      int k = 0;
      for (auto d : box) {
        if (h == d)
          continue;
        if ((d - h).cross(rCenter - h) < 0)
          k++;
      }
      switch (k) {
      case 0:
#ifdef SHOW_IMG
        cv::putText(original, "LB", h, 1, 5, cv::Scalar(255, 0, 100));
#endif
        sortedBox[3] = h;
        break;
      case 1:
#ifdef SHOW_IMG
        cv::putText(original, "LT", h, 1, 5, cv::Scalar(255, 0, 100));
#endif
        sortedBox[0] = h;
        break;
      case 2:
#ifdef SHOW_IMG
        cv::putText(original, "RT", h, 1, 5, cv::Scalar(255, 0, 100));
#endif
        sortedBox[1] = h;
        break;
      case 3:
#ifdef SHOW_IMG
        cv::putText(original, "RB", h, 1, 5, cv::Scalar(255, 0, 100));
#endif
        sortedBox[2] = h;
        break;
      }
    }
    std::vector<cv::Point2f> camPos;
    for (int i = 0; i < 4; i++) {
      float minDis = 1e9;
      cv::Point point2push;
      for (auto p : convexFilted) {
        float d = pow(p.x - sortedBox[i].x, 2) + pow(p.y - sortedBox[i].y, 2);
        if (d < minDis) {
          minDis = d;
          point2push = p;
        }
      }
      camPos.push_back(point2push);
      cv::circle(original, point2push, 10, cv::Scalar(100, 255, 155));
    }
#ifdef SHOW_IMG
    cv::imshow("original", original);
    cv::waitKey(10);
#endif
  }

  cv::Mat original;
  cv::Point3f worldpoints[4]{
      cv::Point3f(-125, 225, 0), cv::Point3f(125, 125, 0),
      cv::Point3f(125, -125, 0), cv::Point3f(-125, -125, 0)};
  cv::Point2f campoints[4]{};
  Eigen::Quaternionf rotate;
  Eigen::Vector3f position;
  std::queue<Eigen::Quaternionf> filter;
  PNPSolver pnpsolver;
};
} // namespace EngineerVisual