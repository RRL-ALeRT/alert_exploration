#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time
from rclpy.duration import Duration
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Quaternion
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from mbf_msgs.action import MoveBase
import math
import numpy as np
import tf2_ros

from alert_exploration.utils import PointCloudFrontierDetector

CLOUD_TOPIC = '/navigation/octomap_point_cloud_centers'
ACTION_SERVER_NAME = '/move_base_flex/move_base'
MAP_FRAME = 'map'
ROBOT_FRAME = 'base_footprint'
MIN_GOAL_DISTANCE = 0.5
GOAL_OFFSET_DISTANCE = 0.75  
WAIT_AFTER_SUCCESS = 1.0
NAV_GOAL_TIMEOUT = 30.0

MAX_NO_FRONTIER_TICKS = 10  # Seconds to wait with no goals before triggering Phase 2

class FrontierNavigationManager(Node):
    def __init__(self):
        super().__init__('frontier_navigation_manager')
        
        self.get_logger().info("Initializing 3D Frontier Manager (With Phase 2 Cleanup)...")
        self.detector = PointCloudFrontierDetector()

        self.goal_start_time = None
        self.goal_handle = None
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.cloud_sub = self.create_subscription(
            PointCloud2, CLOUD_TOPIC, self.cloud_callback, qos)
        
        self.marker_pub = self.create_publisher(MarkerArray, '/frontier_debug_markers', 10)

        self.action_client = ActionClient(self, MoveBase, ACTION_SERVER_NAME)
        self.get_logger().info(f"Waiting for '{ACTION_SERVER_NAME}'...")
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Action server not available yet.")
        else:
            self.get_logger().info("Action server connected!")

        self.exploration_timer = self.create_timer(1.0, self.control_loop)
        
        self.latest_goal_coords = None 
        self.robot_pose = None         
        self.is_navigating = False
        self.is_processing = False
        
        # --- MEMORY ---
        self.blacklisted_goals = [] 
        self.visited_goals = []     
        
        # --- STATE MACHINE ---
        self.exploration_phase = 1
        self.no_frontiers_count = 0
        
        self.current_active_goal = None
        self.wait_until = Time(seconds=0, clock_type=self.get_clock().clock_type)

    def trigger_second_pass(self):
        """Activates Cleanup Phase by clearing memory and making parameters more aggressive."""
        self.get_logger().warn("==================================================")
        self.get_logger().warn("PHASE 1 COMPLETE. Initiating Second Pass (Cleanup)")
        self.get_logger().warn("==================================================")
        
        self.exploration_phase = 2
        self.no_frontiers_count = 0
        
        # 1. Clear the visited memory to allow revisiting areas
        self.visited_goals.clear()
        
        # 2. Make the detector more sensitive to small gaps
        current_min_neighbors = self.detector.params.get('min_neighbors', 5)
        self.detector.params['min_neighbors'] = max(3, current_min_neighbors - 2)
        
        # 3. Shrink the visited radius so it requires a tighter inspection
        current_visited_rad = self.detector.params.get('visited_radius', 1.5)
        self.detector.params['visited_radius'] = current_visited_rad * 0.5 
        
        self.get_logger().info(f"New Visited Radius: {self.detector.params['visited_radius']}m")

    def cloud_callback(self, msg):
        if self.is_processing or self.is_navigating: return
        self.is_processing = True

        try:
            trans = self.tf_buffer.lookup_transform(MAP_FRAME, ROBOT_FRAME, Time())
            self.robot_pose = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
        except Exception:
            self.is_processing = False
            return

        try:
            cloud_generator = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            pts_array = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in cloud_generator], dtype=np.float64)
            
            if pts_array.shape[0] < 10:
                self.is_processing = False
                return

            goal, markers = self.detector.process_pointcloud(
                pts_array, 
                self.robot_pose, 
                blacklist=self.blacklisted_goals,
                visited=self.visited_goals
            )

            self.marker_pub.publish(markers)
            self.latest_goal_coords = goal
            
        except Exception as e:
            self.get_logger().error(f"Error in perception callback: {e}")
        finally:
            self.is_processing = False

    def control_loop(self):

        if self.is_navigating and self.goal_start_time is not None:
            elapsed = (self.get_clock().now() - self.goal_start_time).nanoseconds / 1e9
            if elapsed > NAV_GOAL_TIMEOUT:
                self.get_logger().warn("Navigation goal timed out. Handling as failure.")
                self.abort_current_goal()
                return
        
        if self.is_navigating: return
        
        now = self.get_clock().now()
        if now < self.wait_until: return

        # --- Check for Phase 2 Transition ---
        if self.latest_goal_coords is None:
            self.no_frontiers_count += 1
            
            if self.exploration_phase == 1 and self.no_frontiers_count >= MAX_NO_FRONTIER_TICKS:
                self.trigger_second_pass()
            elif self.exploration_phase == 2 and self.no_frontiers_count >= MAX_NO_FRONTIER_TICKS:
                self.get_logger().info("Map is completely fully explored! Idling...", throttle_duration_sec=10.0)
            else:
                self.get_logger().info("No valid frontiers found, scanning...", throttle_duration_sec=5.0)
            return
        else:
            # Reset counter if we found a goal
            self.no_frontiers_count = 0 

        if self.robot_pose is None: return

        dist = np.linalg.norm(self.latest_goal_coords - self.robot_pose)
        if dist < (MIN_GOAL_DISTANCE + GOAL_OFFSET_DISTANCE): return

        dx = self.latest_goal_coords[0] - self.robot_pose[0]
        dy = self.latest_goal_coords[1] - self.robot_pose[1]
        dist_2d = math.hypot(dx, dy)
        
        offset_x = (dx / dist_2d) * GOAL_OFFSET_DISTANCE
        offset_y = (dy / dist_2d) * GOAL_OFFSET_DISTANCE
        
        safe_goal = np.copy(self.latest_goal_coords)
        safe_goal[0] -= offset_x
        safe_goal[1] -= offset_y

        self.current_active_goal = safe_goal

        goal_point = Point()
        goal_point.x = float(self.current_active_goal[0])
        goal_point.y = float(self.current_active_goal[1])
        goal_point.z = float(self.current_active_goal[2])

        self.get_logger().info(f"[Phase {self.exploration_phase}] Sending Safe Goal: {np.round(self.current_active_goal, 2)}")
        self.send_navigation_goal(goal_point)

    def send_navigation_goal(self, goal_point_3d: Point):
        self.is_navigating = True
        self.goal_start_time = self.get_clock().now()
        
        goal_msg = MoveBase.Goal()
        goal_msg.target_pose.header.frame_id = MAP_FRAME
        goal_msg.target_pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.target_pose.pose.position = goal_point_3d
        
        dx = goal_point_3d.x - self.robot_pose[0]
        dy = goal_point_3d.y - self.robot_pose[1]
        yaw = math.atan2(dy, dx)
        goal_msg.target_pose.pose.orientation = self.yaw_to_quaternion(yaw)

        future = self.action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def abort_current_goal(self):
        if self.goal_handle is not None and self.is_navigating:
            # Only cancel if goal is still active
            if self.goal_handle.status not in [
                GoalStatus.STATUS_SUCCEEDED,
                GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_ABORTED
            ]:
                cancel_future = self.goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(lambda f: self.get_logger().info("Current goal aborted."))
        
        if self.current_active_goal is not None:
            self.blacklisted_goals.append(self.current_active_goal)
        
        self.handle_failure()  

    def yaw_to_quaternion(self, yaw: float) -> Quaternion:
        q = Quaternion()
        q.z = math.sin(yaw / 2.0); q.w = math.cos(yaw / 2.0)
        return q

    def goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().warn('Goal REJECTED.')
            self.handle_failure()
            return
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Goal Reached! Marking area as VISITED.')
            
            if self.current_active_goal is not None:
                self.visited_goals.append(self.current_active_goal)
                if len(self.visited_goals) > 100:
                    self.visited_goals.pop(0)

            self.wait_until = self.get_clock().now() + Duration(seconds=WAIT_AFTER_SUCCESS)
            self.is_navigating = False
            self.latest_goal_coords = None 
            self.current_active_goal = None
        else:
            self.get_logger().warn(f'Navigation Failed (Status: {status}).')
            self.handle_failure()

    def handle_failure(self):
        if self.current_active_goal is not None:
            self.get_logger().info(f"Blacklisting failed goal: {self.current_active_goal}")
            self.blacklisted_goals.append(self.current_active_goal)
            if len(self.blacklisted_goals) > 50: 
                self.blacklisted_goals.pop(0)
        
        self.latest_goal_coords = None 
        self.current_active_goal = None
        self.goal_start_time = None
        self.is_navigating = False

def main(args=None):
    rclpy.init(args=args)
    node = FrontierNavigationManager()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()