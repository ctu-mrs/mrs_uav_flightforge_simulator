#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>

#include <mrs_uav_testing/test_generic.h>

using namespace std::chrono_literals;

class Tester : public mrs_uav_testing::TestGeneric {

public:
  Tester() : mrs_uav_testing::TestGeneric() {
  }

  bool test(void);

  std::shared_ptr<mrs_uav_testing::UAVHandler> uh_;
};

bool Tester::test(void) {

  const std::string uav_name = "uav1";

  {
    auto [uhopt, message] = getUAVHandler(uav_name);

    if (!uhopt) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to obtain handler for '%s': '%s'", uav_name.c_str(), message.c_str());
      return false;
    }

    uh_ = uhopt.value();
  }

  RCLCPP_INFO(node_->get_logger(), "UAVHandler obtained. Initiating takeoff...");

  {
    auto [success, message] = uh_->takeoff();

    if (!success) {
      RCLCPP_ERROR(node_->get_logger(), "Takeoff failed with message: '%s'", message.c_str());
      return false;
    }
  }

  RCLCPP_INFO(node_->get_logger(), "Takeoff succeeded. Hovering for 5 seconds...");
  sleep(5.0);

  if (!uh_->isFlyingNormally()) {
    RCLCPP_ERROR(node_->get_logger(), "UAV is not flying normally!");
    return false;
  }

  RCLCPP_INFO(node_->get_logger(), "UAV is flying normally. Initiating landing...");

  {
    auto [success, message] = uh_->land();

    if (!success) {
      RCLCPP_ERROR(node_->get_logger(), "Landing failed with message: '%s'", message.c_str());
      return false;
    }
  }

  RCLCPP_INFO(node_->get_logger(), "Takeoff and landing test finished successfully!");
  return true;
}

int main(int argc, char *argv[]) {

  rclcpp::init(argc, argv);

  bool test_result = true;

  Tester tester;

  test_result &= tester.test();

  tester.reportTestResult(test_result);

  tester.join();
}

