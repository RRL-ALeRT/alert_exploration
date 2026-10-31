#!/usr/bin/env python3

import math

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, Quaternion
from mbf_msgs.action import MoveBase
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros


class RedSurfaceMBFNavigator(Node):
    def __init__(self):
        super().__init__('red_surface_mbf_navigator')

        self.declare_parameter('candidate_topic', '/red_surface_candidate_goal')
        self.declare_parameter('action_server', '/move_base_flex/move_base')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('goal_timeout', 90.0)
        self.declare_parameter('minimum_goal_separation', 0.75)

        candidate_topic = self.get_parameter('candidate_topic').value
        action_server = self.get_parameter('action_server').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.goal_timeout = float(self.get_parameter('goal_timeout').value)
        self.minimum_goal_separation = float(
            self.get_parameter('minimum_goal_separation').value)

        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.subscription = self.create_subscription(
            PointStamped, candidate_topic, self.candidate_callback, qos)
        self.action_client = ActionClient(self, MoveBase, action_server)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.goal_handle = None
        self.active_goal = None
        self.goal_start_time = None
        self.timer = self.create_timer(1.0, self.watchdog)

        self.get_logger().info(
            f'Waiting for MBF action server {action_server} and candidate topic {candidate_topic}')
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('MBF action server is not available yet.')
        else:
            self.get_logger().info('MBF action server connected.')

    def candidate_callback(self, message):
        if self.active_goal is not None:
            return

        candidate = message.point
        if not all(math.isfinite(value) for value in (
                candidate.x, candidate.y, candidate.z)):
            self.get_logger().warn('Ignoring non-finite candidate goal.')
            return

        robot_xy = self._robot_xy(message.header.frame_id)
        if robot_xy is None:
            self.get_logger().warn('Cannot transform robot pose; ignoring candidate.')
            return

        distance = math.hypot(candidate.x - robot_xy[0], candidate.y - robot_xy[1])
        if distance < self.minimum_goal_separation:
            self.get_logger().warn(
                f'Ignoring candidate only {distance:.2f} m from robot.')
            return

        yaw = math.atan2(candidate.y - robot_xy[1], candidate.x - robot_xy[0])
        goal = MoveBase.Goal()
        goal.target_pose.header = message.header
        goal.target_pose.header.stamp = self.get_clock().now().to_msg()
        goal.target_pose.pose.position = candidate
        goal.target_pose.pose.orientation = self._yaw_to_quaternion(yaw)

        self.active_goal = candidate
        self.goal_start_time = self.get_clock().now()
        self.get_logger().info(
            f'Sending MBF goal [{candidate.x:.2f}, {candidate.y:.2f}, {candidate.z:.2f}]')
        future = self.action_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)

    def _robot_xy(self, frame_id):
        try:
            transform = self.tf_buffer.lookup_transform(
                frame_id, self.robot_frame, Time())
            return (
                transform.transform.translation.x,
                transform.transform.translation.y)
        except Exception as error:
            self.get_logger().debug(f'Robot transform unavailable: {error}')
            return None

    def goal_response_callback(self, future):
        try:
            self.goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(f'MBF goal request failed: {error}')
            self._clear_goal()
            return

        if not self.goal_handle.accepted:
            self.get_logger().warn('MBF rejected the candidate goal.')
            self._clear_goal()
            return

        self.get_logger().info('MBF accepted the candidate goal.')
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('MBF reached the red-surface candidate.')
        else:
            self.get_logger().warn(f'MBF finished with status {status}.')
        self._clear_goal()

    def watchdog(self):
        if self.active_goal is None or self.goal_start_time is None:
            return
        elapsed = (self.get_clock().now() - self.goal_start_time).nanoseconds / 1e9
        if elapsed <= self.goal_timeout:
            return

        self.get_logger().warn('MBF goal timed out; cancelling it.')
        if self.goal_handle is not None:
            future = self.goal_handle.cancel_goal_async()
            future.add_done_callback(lambda _: self.get_logger().info('MBF goal cancelled.'))
        self._clear_goal()

    def _clear_goal(self):
        self.goal_handle = None
        self.active_goal = None
        self.goal_start_time = None

    @staticmethod
    def _yaw_to_quaternion(yaw):
        quaternion = Quaternion()
        quaternion.z = math.sin(yaw / 2.0)
        quaternion.w = math.cos(yaw / 2.0)
        return quaternion


def main(args=None):
    rclpy.init(args=args)
    node = RedSurfaceMBFNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
