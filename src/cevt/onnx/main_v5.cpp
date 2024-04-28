#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>

// #include <inference_engine.hpp>
#include <openvino/openvino.hpp>

using namespace cv;
using namespace dnn;

//??????
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

class FPSCounter {
public:
  bool Count() {
    if (_count == 0) {
      _count = 1;
      _timingStart = std::chrono::steady_clock::now();
    } else {
      ++_count;
      if (std::chrono::steady_clock::now() - _timingStart >=
          std::chrono::seconds(1)) {
        _lastFPS = _count;
        _count = 0;
        return true;
      }
    }
    return false;
  }

  int GetFPS() { return _lastFPS; }

private:
  int _count = 0, _lastFPS;
  std::chrono::steady_clock::time_point _timingStart;
};

int main() {
  // step1:?????
  auto start = std::chrono::steady_clock::now();
  ov::Core core;
  // step2:??????
  auto compiled_model =
      core.compile_model("./models/buff_nocolor_v4.onnx", "GPU");
  ov::InferRequest infer_request = compiled_model.create_infer_request();
  auto end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  if (compiled_model) {
    std::cout << "模型加载成功，耗时" << duration.count() << "ms"
              << std::endl; //???2s?????????????openvino?????
  } else {
    std::cout << "模型加载失败";
  }
  cv::VideoCapture vid("videos/test_vid.mp4");
  std::cout << vid.isOpened() << std::endl;
  cv::Mat img;

  int font_face = cv::FONT_HERSHEY_SIMPLEX;
  double font_scale = 1;
  int thickness = 2;
  int baseline = 0;
  int padding = 10;
  cv::Scalar text_color(255, 255, 255);
  auto fps = FPSCounter();
  std::string fpsMessage = "FPS:";
  while (true) {
    // step3??4:?????????????
    auto start2 = std::chrono::steady_clock::now();
    // cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    vid.read(img);
    std::cout << img.empty() << std::endl;
    cv::Mat letterbox_img = letterbox(img);
    float scale = letterbox_img.size[0] / 352.0;
    cv::Mat blob = cv::dnn::blobFromImage(
        letterbox_img, 1.0 / 255.0, cv::Size(352, 352), cv::Scalar(), true);
    auto input_port = compiled_model.input();
    // std::cout << "step 3 completed" << std::endl;
    //  step5:??????????

    ov::Tensor input_tensor(input_port.get_element_type(),
                            input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor);

    // -------- Step 6. Start inference --------
    infer_request.infer();
    // std::cout << "step 6 completed" << std::endl;
    //  -------- Step 7. Get the inference result --------
    auto output = infer_request.get_output_tensor(0);
    auto output_shape = output.get_shape();
    // std::cout << "The shape of output tensor:" << output_shape << std::endl;
    //[8400,25]??????8400??anchor?????anchor????25???????????x,y,w,h,??????????????????????????????????x,y,???????

    float *data = output.data<float>();
    Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
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
    std::cout << "置信度为：" << ToShoot.at<float>(0, 5) << std::endl;
    float cxR = R.at<float>(0, 0);
    float cyR = R.at<float>(0, 1);
    float wR = R.at<float>(0, 2);
    float hR = R.at<float>(0, 3);
    int leftR = int((cxR - 0.5 * wR) * scale);
    int topR = int((cyR - 0.5 * hR) * scale);
    int widthR = int(wR * scale);
    int heightR = int(hR * scale);
    cv::rectangle(img, Rect(leftR, topR, widthR, heightR), Scalar(0, 0, 255),
                  1);
    if (confidence[1] > 0.25) {
      float cx = ToShoot.at<float>(0, 0);
      float cy = ToShoot.at<float>(0, 1);
      float w = ToShoot.at<float>(0, 2);
      float h = ToShoot.at<float>(0, 3);
      int left = int((cx - 0.5 * w) * scale);
      int top = int((cy - 0.5 * h) * scale);
      int width = int(w * scale);
      int height = int(h * scale);
      cv::rectangle(img, Rect(left, top, width, height), Scalar(0, 0, 255), 1);
      for (int j = 0; j < 5; j++) {
        float x = ToShoot.at<float>(0, 7 + j * 3) * scale;
        float y = ToShoot.at<float>(0, 7 + j * 3 + 1) * scale;
        circle(img, Point(x, y), 10, colors[j], -1);
      }
    } else {
      std::cout << "no to shoot" << std::endl;
    }
    auto end2 = std::chrono::steady_clock::now();
    auto duration2 =
        std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    std::cout << "运行时间：" << duration2.count() << "ms" << std::endl;

    if (fps.Count()) {
      std::cout << "FPS:" << fps.GetFPS() << std::endl;
    }
    fpsMessage += std::to_string(fps.GetFPS());
    putText(img, fpsMessage, Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 1.5,
            cv::Scalar(0, 0, 255), 2);
    namedWindow("Demo", WINDOW_AUTOSIZE);
    imshow("Demo", img);
    moveWindow("Demo", 100, 100);
    waitKey(100);
    fpsMessage.clear();
  }

  return 0;
}
