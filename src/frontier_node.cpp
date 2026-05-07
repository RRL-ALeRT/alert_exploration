#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/time.h>
#include <pcl_conversions/pcl_conversions.h>
#include <mbf_msgs/action/move_base.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "alert_exploration/frontier_detector.hpp"

using std::placeholders::_1;

class FrontierNode : public rclcpp::Node {
public:
  using MoveBase = mbf_msgs::action::MoveBase;
  using GoalHandleMoveBase = rclcpp_action::ClientGoalHandle<MoveBase>;

  FrontierNode()
  : Node("frontier_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_) {
    cloud_topic_ = this->declare_parameter<std::string>("cloud_topic", "/octomap_point_cloud_centers");
    octomap_topic_ = this->declare_parameter<std::string>("octomap_topic", "/octomap_binary");
    action_server_name_ = this->declare_parameter<std::string>("action_server", "/move_base_flex/move_base");
    map_frame_ = this->declare_parameter<std::string>("map_frame", "map");
    robot_frame_ = this->declare_parameter<std::string>("robot_frame", "base_footprint");
    robot_frame_fallback_ = this->declare_parameter<std::string>("robot_frame_fallback", "base_link");
    use_octomap_ = this->declare_parameter<bool>("use_octomap", true);
    if (use_octomap_) {
      octomap_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
        octomap_topic_, rclcpp::QoS(1).best_effort(), std::bind(&FrontierNode::octomap_cb, this, _1));
    } else {
      cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, rclcpp::SensorDataQoS(), std::bind(&FrontierNode::cloud_cb, this, _1));
    }
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/frontier_debug_markers", 10);
    action_client_ = rclcpp_action::create_client<MoveBase>(this, action_server_name_);
    detector_ = std::make_shared<FrontierDetector>();
    control_timer_ = this->create_wall_timer(std::chrono::seconds(1), std::bind(&FrontierNode::control_loop, this));
    wait_until_ = this->get_clock()->now();
    last_log_time_ = this->get_clock()->now();
    last_input_time_ = this->get_clock()->now();

    if (use_octomap_) {
      RCLCPP_INFO(this->get_logger(), "Input mode: octomap (%s)", octomap_topic_.c_str());
    } else {
      RCLCPP_INFO(this->get_logger(), "Input mode: pointcloud (%s)", cloud_topic_.c_str());
    }
    RCLCPP_INFO(this->get_logger(), "TF frames: map=%s robot=%s fallback=%s",
      map_frame_.c_str(), robot_frame_.c_str(), robot_frame_fallback_.c_str());

    RCLCPP_INFO(this->get_logger(), "FrontierNode started, waiting for %s", action_server_name_.c_str());
    if (!action_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_WARN(this->get_logger(), "Action server not available yet.");
    } else {
      RCLCPP_INFO(this->get_logger(), "Action server connected.");
    }
  }

private:
  static double sq_dist(const pcl::PointXYZ &a, const pcl::PointXYZ &b) {
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    const double dz = static_cast<double>(a.z) - static_cast<double>(b.z);
    return dx * dx + dy * dy + dz * dz;
  }

  geometry_msgs::msg::Quaternion yaw_to_quaternion(double yaw) const {
    geometry_msgs::msg::Quaternion q;
    q.z = std::sin(yaw / 2.0);
    q.w = std::cos(yaw / 2.0);
    return q;
  }

  bool near_any_goal(const pcl::PointXYZ &goal, const std::vector<pcl::PointXYZ> &goals, double radius) const {
    const double radius_sq = radius * radius;
    for (const auto &g : goals) {
      if (sq_dist(goal, g) < radius_sq) {
        return true;
      }
    }
    return false;
  }

  bool update_robot_pose_from_tf() {
    auto fill_pose = [&](const geometry_msgs::msg::TransformStamped &trans) {
      robot_pose_.x = static_cast<float>(trans.transform.translation.x);
      robot_pose_.y = static_cast<float>(trans.transform.translation.y);
      robot_pose_.z = static_cast<float>(trans.transform.translation.z);
      robot_pose_valid_ = true;
    };

    try {
      const auto trans = tf_buffer_.lookupTransform(map_frame_, robot_frame_, tf2::TimePointZero);
      fill_pose(trans);
      using_fallback_robot_frame_ = false;
      return true;
    } catch (const std::exception &) {
      if (!robot_frame_fallback_.empty()) {
        try {
          const auto trans = tf_buffer_.lookupTransform(map_frame_, robot_frame_fallback_, tf2::TimePointZero);
          fill_pose(trans);
          if (!using_fallback_robot_frame_) {
            RCLCPP_WARN(this->get_logger(), "Using fallback robot frame '%s' (primary '%s' unavailable).",
              robot_frame_fallback_.c_str(), robot_frame_.c_str());
          }
          using_fallback_robot_frame_ = true;
          return true;
        } catch (const std::exception &) {
          robot_pose_valid_ = false;
          return false;
        }
      }

      robot_pose_valid_ = false;
      return false;
    }
  }

  void publish_markers(
    const std::vector<FrontierCandidate> &raw,
    const std::vector<FrontierCandidate> &filtered,
    bool has_best,
    const FrontierCandidate &best,
    const std_msgs::msg::Header &header) {
    visualization_msgs::msg::MarkerArray ma;

    visualization_msgs::msg::Marker clear;
    clear.header = header;
    clear.ns = "frontier_debug";
    clear.id = 0;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    ma.markers.push_back(clear);

    auto make_sphere_list = [&](int id, const std::string &ns, float r, float g, float b, float a, float scale) {
      visualization_msgs::msg::Marker m;
      m.header = header;
      m.ns = ns;
      m.id = id;
      m.type = visualization_msgs::msg::Marker::SPHERE_LIST;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.scale.x = scale;
      m.scale.y = scale;
      m.scale.z = scale;
      m.color.r = r;
      m.color.g = g;
      m.color.b = b;
      m.color.a = a;
      return m;
    };

    if (!raw.empty()) {
      auto m = make_sphere_list(1, "raw_goals", 1.0f, 0.0f, 0.0f, 0.35f, 0.2f);
      for (const auto &c : raw) {
        geometry_msgs::msg::Point p;
        p.x = c.medoid.x;
        p.y = c.medoid.y;
        p.z = c.medoid.z;
        m.points.push_back(p);
      }
      ma.markers.push_back(m);
    }

    if (!filtered.empty()) {
      auto m = make_sphere_list(2, "filtered_goals", 1.0f, 1.0f, 0.0f, 0.7f, 0.3f);
      for (const auto &c : filtered) {
        geometry_msgs::msg::Point p;
        p.x = c.medoid.x;
        p.y = c.medoid.y;
        p.z = c.medoid.z;
        m.points.push_back(p);
      }
      ma.markers.push_back(m);
    }

    if (has_best) {
      visualization_msgs::msg::Marker best_m;
      best_m.header = header;
      best_m.ns = "best_goal";
      best_m.id = 3;
      best_m.type = visualization_msgs::msg::Marker::SPHERE;
      best_m.action = visualization_msgs::msg::Marker::ADD;
      best_m.scale.x = 0.5;
      best_m.scale.y = 0.5;
      best_m.scale.z = 0.5;
      best_m.color.r = 0.0f;
      best_m.color.g = 1.0f;
      best_m.color.b = 0.0f;
      best_m.color.a = 1.0f;
      best_m.pose.position.x = best.medoid.x;
      best_m.pose.position.y = best.medoid.y;
      best_m.pose.position.z = best.medoid.z;
      ma.markers.push_back(best_m);
    }

    marker_pub_->publish(ma);
  }

  void process_cloud_and_update(const PointCloudT::Ptr &cloud, const std_msgs::msg::Header &header) {
    if (is_processing_ || is_navigating_) {
      return;
    }
    is_processing_ = true;

    do {
      if (!update_robot_pose_from_tf()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
          "TF unavailable for map '%s' and robot frames '%s'/'%s'.",
          map_frame_.c_str(), robot_frame_.c_str(), robot_frame_fallback_.c_str());
        break;
      }

      if (cloud->points.size() < 10) {
        break;
      }

      auto raw_candidates = detector_->detect(cloud);
      std::vector<FrontierCandidate> filtered;
      filtered.reserve(raw_candidates.size());

      for (const auto &cand : raw_candidates) {
        if (std::fabs(cand.medoid.z - robot_pose_.z) > max_height_diff_) {
          continue;
        }

        if (std::sqrt(sq_dist(cand.medoid, robot_pose_)) < min_dist_from_robot_) {
          continue;
        }

        if (near_any_goal(cand.medoid, visited_goals_, visited_radius_)) {
          continue;
        }

        if (near_any_goal(cand.medoid, blacklisted_goals_, blacklist_radius_)) {
          continue;
        }

        filtered.push_back(cand);
      }

      bool has_best = false;
      FrontierCandidate best;
      if (!filtered.empty()) {
        double max_dist = std::numeric_limits<double>::lowest();
        int max_size = 1;

        std::vector<double> dists;
        dists.reserve(filtered.size());
        for (const auto &cand : filtered) {
          const double d = std::sqrt(sq_dist(cand.medoid, robot_pose_));
          dists.push_back(d);
          if (d > max_dist) {
            max_dist = d;
          }
          if (cand.size > max_size) {
            max_size = cand.size;
          }
        }

        const double size_weight = 1.5;
        const double distance_weight = 1.0;
        double best_utility = std::numeric_limits<double>::lowest();
        size_t best_index = 0;
        const double safe_max_dist = max_dist > 0.0 ? max_dist : 1.0;

        for (size_t i = 0; i < filtered.size(); ++i) {
          const double norm_dist = dists[i] / safe_max_dist;
          const double norm_size = static_cast<double>(filtered[i].size) / static_cast<double>(max_size);
          const double utility = (size_weight * norm_size) - (distance_weight * norm_dist);
          if (utility > best_utility) {
            best_utility = utility;
            best_index = i;
          }
        }

        best = filtered[best_index];
        latest_goal_ = best.medoid;
        latest_goal_valid_ = true;
        has_best = true;
      } else {
        latest_goal_valid_ = false;
      }

      publish_markers(raw_candidates, filtered, has_best, best, header);

      if (has_best) {
        const double pos_eps = 0.15;
        const int size_eps = 5;
        const double unchanged_log_period_s = 2.0;

        const bool changed =
          !has_last_goal_ ||
          std::fabs(best.medoid.x - last_goal_.x) > pos_eps ||
          std::fabs(best.medoid.y - last_goal_.y) > pos_eps ||
          std::fabs(best.medoid.z - last_goal_.z) > pos_eps ||
          std::abs(best.size - last_goal_size_) >= size_eps;

        const auto now = this->get_clock()->now();
        const bool periodic = (now - last_log_time_).seconds() >= unchanged_log_period_s;

        if (changed || periodic) {
          RCLCPP_INFO(this->get_logger(), "Best Goal: [%.2f, %.2f, %.2f] size=%d",
            best.medoid.x, best.medoid.y, best.medoid.z, best.size);
          last_log_time_ = now;
        }

        last_goal_ = best.medoid;
        last_goal_size_ = best.size;
        has_last_goal_ = true;
      }
    } while (false);

    is_processing_ = false;
  }

  void cloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    has_input_ = true;
    last_input_time_ = this->get_clock()->now();
    PointCloudT::Ptr cloud(new PointCloudT());
    pcl::fromROSMsg(*msg, *cloud);
    process_cloud_and_update(cloud, msg->header);
  }

  void octomap_cb(const octomap_msgs::msg::Octomap::SharedPtr msg) {
    has_input_ = true;
    last_input_time_ = this->get_clock()->now();
    octomap::AbstractOcTree *tree = nullptr;
    if (msg->binary) {
      tree = octomap_msgs::binaryMsgToMap(*msg);
    } else {
      tree = octomap_msgs::fullMsgToMap(*msg);
    }

    if (!tree) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000, "Failed to convert Octomap message.");
      return;
    }

    auto *octree = dynamic_cast<octomap::OcTree *>(tree);
    if (!octree) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000, "Converted Octomap is not an OcTree.");
      delete tree;
      return;
    }

    PointCloudT::Ptr cloud(new PointCloudT());
    cloud->points.reserve(octree->size());
    for (auto it = octree->begin_leafs(), end = octree->end_leafs(); it != end; ++it) {
      if (octree->isNodeOccupied(*it)) {
        cloud->points.emplace_back(static_cast<float>(it.getX()), static_cast<float>(it.getY()), static_cast<float>(it.getZ()));
      }
    }
    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = true;

    process_cloud_and_update(cloud, msg->header);
    delete tree;
  }

  void send_navigation_goal(const pcl::PointXYZ &goal) {
    if (!action_client_->action_server_is_ready()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000, "Action server unavailable.");
      return;
    }

    MoveBase::Goal goal_msg;
    goal_msg.target_pose.header.frame_id = map_frame_;
    goal_msg.target_pose.header.stamp = this->get_clock()->now();
    goal_msg.target_pose.pose.position.x = goal.x;
    goal_msg.target_pose.pose.position.y = goal.y;
    goal_msg.target_pose.pose.position.z = goal.z;

    const double dx = static_cast<double>(goal.x) - static_cast<double>(robot_pose_.x);
    const double dy = static_cast<double>(goal.y) - static_cast<double>(robot_pose_.y);
    const double yaw = std::atan2(dy, dx);
    goal_msg.target_pose.pose.orientation = yaw_to_quaternion(yaw);

    rclcpp_action::Client<MoveBase>::SendGoalOptions options;
    options.goal_response_callback = std::bind(&FrontierNode::goal_response_callback, this, std::placeholders::_1);
    options.result_callback = std::bind(&FrontierNode::goal_result_callback, this, std::placeholders::_1);

    is_navigating_ = true;
    goal_start_time_ = this->get_clock()->now();
    action_client_->async_send_goal(goal_msg, options);
  }

  void goal_response_callback(const GoalHandleMoveBase::SharedPtr &goal_handle) {
    goal_handle_ = goal_handle;
    if (!goal_handle_) {
      RCLCPP_WARN(this->get_logger(), "Goal rejected.");
      handle_failure();
    }
  }

  void goal_result_callback(const GoalHandleMoveBase::WrappedResult &result) {
    if (result.code == rclcpp_action::ResultCode::SUCCEEDED &&
      result.result &&
      result.result->outcome == MoveBase::Result::SUCCESS)
    {
      RCLCPP_INFO(this->get_logger(), "Goal reached. Marking area as visited.");
      if (current_active_goal_valid_) {
        visited_goals_.push_back(current_active_goal_);
        if (visited_goals_.size() > 100) {
          visited_goals_.erase(visited_goals_.begin());
        }
      }
      wait_until_ = this->get_clock()->now() + rclcpp::Duration::from_seconds(wait_after_success_s_);
      is_navigating_ = false;
      latest_goal_valid_ = false;
      current_active_goal_valid_ = false;
      goal_handle_.reset();
      return;
    }

    const uint32_t outcome = result.result ? result.result->outcome : 999;
    RCLCPP_WARN(this->get_logger(), "Navigation failed (result code=%d, outcome=%u).",
      static_cast<int>(result.code), outcome);
    handle_failure();
  }

  void abort_current_goal() {
    if (goal_handle_ && is_navigating_) {
      action_client_->async_cancel_goal(goal_handle_);
    }
    handle_failure();
  }

  void handle_failure() {
    if (current_active_goal_valid_) {
      if (!near_any_goal(current_active_goal_, blacklisted_goals_, blacklist_radius_)) {
        blacklisted_goals_.push_back(current_active_goal_);
        if (blacklisted_goals_.size() > 50) {
          blacklisted_goals_.erase(blacklisted_goals_.begin());
        }
      }
    }

    latest_goal_valid_ = false;
    current_active_goal_valid_ = false;
    is_navigating_ = false;
    goal_handle_.reset();
  }

  void control_loop() {
    const auto now = this->get_clock()->now();

    if (!has_input_) {
      if (use_octomap_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
          "No messages received on octomap topic '%s'.", octomap_topic_.c_str());
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
          "No messages received on pointcloud topic '%s'.", cloud_topic_.c_str());
      }
      return;
    }

    if ((now - last_input_time_).seconds() > 3.0) {
      if (use_octomap_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
          "Octomap topic '%s' is stale (>3s).", octomap_topic_.c_str());
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
          "Pointcloud topic '%s' is stale (>3s).", cloud_topic_.c_str());
      }
    }

    if (is_navigating_ && goal_start_time_.nanoseconds() > 0) {
      const double elapsed = (now - goal_start_time_).seconds();
      if (elapsed > nav_goal_timeout_s_) {
        RCLCPP_WARN(this->get_logger(), "Navigation goal timed out.");
        abort_current_goal();
        return;
      }
    }

    if (is_navigating_) {
      return;
    }

    if (now < wait_until_) {
      return;
    }

    if (!latest_goal_valid_) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "No valid frontiers found, scanning...");
      return;
    }

    if (!robot_pose_valid_) {
      return;
    }

    const double dist = std::sqrt(sq_dist(latest_goal_, robot_pose_));
    if (dist < (min_goal_distance_ + goal_offset_distance_)) {
      return;
    }

    const double dx = static_cast<double>(latest_goal_.x) - static_cast<double>(robot_pose_.x);
    const double dy = static_cast<double>(latest_goal_.y) - static_cast<double>(robot_pose_.y);
    const double dist_2d = std::hypot(dx, dy);
    if (dist_2d < 1e-6) {
      latest_goal_valid_ = false;
      return;
    }

    pcl::PointXYZ safe_goal = latest_goal_;
    safe_goal.x -= static_cast<float>((dx / dist_2d) * goal_offset_distance_);
    safe_goal.y -= static_cast<float>((dy / dist_2d) * goal_offset_distance_);

    current_active_goal_ = safe_goal;
    current_active_goal_valid_ = true;

    RCLCPP_INFO(this->get_logger(), "Sending safe goal: [%.2f, %.2f, %.2f]",
      safe_goal.x, safe_goal.y, safe_goal.z);
    send_navigation_goal(safe_goal);
  }

  // interfaces
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp_action::Client<MoveBase>::SharedPtr action_client_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::shared_ptr<FrontierDetector> detector_;

  // state
  bool is_processing_{false};
  bool is_navigating_{false};
  bool robot_pose_valid_{false};
  bool latest_goal_valid_{false};
  bool current_active_goal_valid_{false};
  bool has_last_goal_{false};
  pcl::PointXYZ robot_pose_{};
  pcl::PointXYZ latest_goal_{};
  pcl::PointXYZ current_active_goal_{};
  pcl::PointXYZ last_goal_{};
  int last_goal_size_{0};
  rclcpp::Time last_log_time_;
  rclcpp::Time goal_start_time_;
  rclcpp::Time wait_until_;
  GoalHandleMoveBase::SharedPtr goal_handle_;
  std::vector<pcl::PointXYZ> blacklisted_goals_;
  std::vector<pcl::PointXYZ> visited_goals_;

  // config
  std::string cloud_topic_;
  std::string octomap_topic_;
  std::string action_server_name_;
  std::string map_frame_;
  std::string robot_frame_;
  std::string robot_frame_fallback_;
  const double min_goal_distance_ = 0.5;
  const double goal_offset_distance_ = 0.75;
  const double min_dist_from_robot_ = 0.8;
  const double max_height_diff_ = 1.5;
  const double visited_radius_ = 1.5;
  const double blacklist_radius_ = 1.0;
  const double wait_after_success_s_ = 1.0;
  const double nav_goal_timeout_s_ = 90.0;
  bool use_octomap_{true};
  bool has_input_{false};
  bool using_fallback_robot_frame_{false};
  rclcpp::Time last_input_time_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FrontierNode>());
  rclcpp::shutdown();
  return 0;
}
