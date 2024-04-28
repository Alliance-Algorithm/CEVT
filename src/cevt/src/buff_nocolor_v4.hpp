#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <string>
#include <vector>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include <openvino/openvino.hpp>
class buff_nocolor {
  std::vector<cv::Scalar> colors = {
      cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 255), cv::Scalar(170, 0, 255),
      cv::Scalar(255, 0, 85), cv::Scalar(255, 0, 170)};

  cv::Mat letterbox(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
  }

  void normalizeArray(float arr[], int size) {
    // ???????????????
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
      sum += arr[i];
    }

    // ????????????
    for (int i = 0; i < size; i++) {
      arr[i] /= sum;
    }
  }

  int findMaxIndex(float arr[], int size) {
    float maxVal = arr[0];
    int maxIndex = 0;

    for (int i = 1; i < size; i++) {
      if (arr[i] > maxVal) {
        maxVal = arr[i];
        maxIndex = i;
      }
    }
    return maxIndex;
  }

  float findMaxValue(float arr[], int size) {
    float maxVal = arr[0];

    for (int i = 1; i < size; i++) {
      if (arr[i] > maxVal) {
        maxVal = arr[i];
      }
    }

    return maxVal;
  }

public:
  buff_nocolor() {

    // step1:?????
    ov::Core core;
    // step2:??????
    compile_model_ = core.compile_model(
        "/workspaces/CEVT/models/buff_nocolor_v4.onnx", "CPU");
    infer_request_ = compile_model_.create_infer_request();

    if (compile_model_) {
      std::cout << "模型加载成功"
                << std::endl; //???2s?????????????openvino?????
    } else {
      std::cout << "模型加载失败";
    }
    cv::Mat img;
    cv::Scalar text_color(255, 255, 255);
    std::string fpsMessage = "FPS:";
  }

  bool Calculate(cv::Mat &img, std::vector<cv::Point2f> &cors) {
    cors.clear();
    // cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::Mat letterbox_img = letterbox(img);
    float scale = letterbox_img.size[0] / 352.0;
    cv::Mat blob = cv::dnn::blobFromImage(
        letterbox_img, 1.0 / 255.0, cv::Size(352, 352), cv::Scalar(), true);
    auto input_port = compile_model_.input();
    // std::cout << "step 3 completed" << std::endl;
    //  step5:??????????

    ov::Tensor input_tensor(input_port.get_element_type(),
                            input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request_.set_input_tensor(input_tensor);

    // -------- Step 6. Start inference --------
    infer_request_.infer();
    // std::cout << "step 6 completed" << std::endl;
    //  -------- Step 7. Get the inference result --------
    auto output = infer_request_.get_output_tensor(0);
    auto output_shape = output.get_shape();
    // std::cout << "The shape of output tensor:" << output_shape << std::endl;
    //[8400,25]??????8400??anchor?????anchor????25???????????x,y,w,h,??????????????????????????????????x,y,???????

    float *data = output.data<float>();
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer);

    float confidence[3] = {.0f, .0f, .0f}; //?????????????
    int confidence_index[3] = {0, 0, 0};   //??????????????��??

    //???????????????????????????????anchor
    for (int i = 0; i < output_buffer.rows; ++i) {
      float conf[3];
      for (int j = 4; j < 7; ++j) {
        conf[j - 4] = output_buffer.at<float>(i, j);
      }
      /*int num = findMaxIndex(conf, 3);
      if (num == 0)
      {
              normalizeArray(conf, 3);
      }
      for (int ind = 4; ind < 7; ind++)
      {
              output_buffer.at<float>(i, ind) = conf[ind - 4];
      }*/
      int category = findMaxIndex(conf, 3);
      if (conf[category] > confidence[category]) {
        confidence[category] = conf[category];
        confidence_index[category] = i;
      }
    }

    //???
    cv::Mat R = output_buffer.row(confidence_index[0]);
    cv::Mat ToShoot = output_buffer.row(confidence_index[1]);
    // cv::rectangle(img, cv::Rect(leftR, topR, widthR, heightR),
    //               cv::Scalar(0, 0, 255), 1);
    if (confidence[1] > 0.25) {
      float cx = ToShoot.at<float>(0, 0);
      float cy = ToShoot.at<float>(0, 1);
      float w = ToShoot.at<float>(0, 2);
      float h = ToShoot.at<float>(0, 3);
      int left = int((cx - 0.5 * w) * scale);
      int top = int((cy - 0.5 * h) * scale);
      int width = int(w * scale);
      int height = int(h * scale);
      cv::rectangle(img, cv::Rect(left, top, width, height),
                    cv::Scalar(0, 0, 255), 1);
      for (int j = 0; j < 5; j++) {
        float x = ToShoot.at<float>(0, 7 + j * 3) * scale;
        float y = ToShoot.at<float>(0, 7 + j * 3 + 1) * scale;
        cors.push_back(cv::Point2f(x, y));
        // circle(img, cv::Point(x, y), 10, colors[j], -1);
      }
    } else {
      std::cout << "no to shoot" << std::endl;
    }

    return true;
  }

  ov::CompiledModel compile_model_;
  ov::InferRequest infer_request_;
};