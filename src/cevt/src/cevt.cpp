#include <cstdio>
#include <geometry_msgs/msg/detail/pose_array__struct.hpp>
#include <memory>
#include <rclcpp/qos.hpp>
#include <thread>

#include <eigen3/Eigen/Dense>

#include <hikcamera/image_capturer.hpp>

#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/detail/marker__struct.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "point_detect.hpp"
#include "util/fps_counter.hpp"

namespace EngineerVisual {
class CameraNode : public rclcpp::Node {

public:
  CameraNode(const std::string &node_name)
      : Node(node_name), points_publisger_(),
        thread_(&CameraNode::thread_main, this) {
    points_publisger_ = create_publisher<geometry_msgs::msg::PoseArray>(
        "/cevt/points", rclcpp::QoS(1));
    marker_publisher_ = create_publisher<visualization_msgs::msg::Marker>(
        "/engineer/points/marker", rclcpp::QoS(1));
  }

private:
  void thread_main() {

    FpsCounter fps_counter;

    hikcamera::ImageCapturer::CameraProfile camera_profile;
    {
      using namespace std::chrono_literals;
      camera_profile.exposure_time = 3ms;
      camera_profile.gain = 16.9807;
      camera_profile.invert_image = true;
    }
    hikcamera::ImageCapturer image_capturer(camera_profile);

    while (rclcpp::ok()) {
      if (fps_counter.count())
        RCLCPP_INFO(this->get_logger(), "fps: %d ", fps_counter.get_fps());

      geometry_msgs::msg::Pose msg;
      visualization_msgs::msg::Marker ore_marker;

      auto image = image_capturer.read();

      detecter.Calculate(image);

      auto rotate = detecter.Rotate();
      auto position = detecter.Position();
      ore_marker.header.frame_id = "engineer";
      ore_marker.type = visualization_msgs::msg::Marker::SPHERE;
      ore_marker.action = visualization_msgs::msg::Marker::ADD;
      ore_marker.scale.x = ore_marker.scale.y = ore_marker.scale.z = 0.05;
      ore_marker.color.r = 1.0;
      ore_marker.color.g = 0.0;
      ore_marker.color.b = 0.0;
      ore_marker.color.a = 1.0;
      ore_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
      ore_marker.header.stamp = now();
      ore_marker.pose.position.x = position.x();
      ore_marker.pose.position.y = position.y();
      ore_marker.pose.position.z = position.z();
      marker_publisher_->publish(ore_marker);

      // points_publisger_->publish(msg);
    }
  }

  PointDetect detecter;

  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr points_publisger_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      marker_publisher_;

  std::thread thread_;
};

} // namespace EngineerVisual

int main(int argc, char **argv) {

  rclcpp::init(argc, argv);
  rclcpp::executors::SingleThreadedExecutor executor;
  auto node = std::make_shared<EngineerVisual::CameraNode>("cevt");
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();

  // Test
  // hikcamera::ImageCapturer hikCapture;
  // auto rawImg = hikCapture.read();

  // cv::imshow("raw", rawImg);

  // cv::waitKey(0); // 这句确保窗口一直打开
  // TestEnd

  printf("hello world cevt package\n");
  return 0;
}
