#!/usr/bin/env python3

import math
import numpy as np
import open3d as o3d
import rclpy
from scipy.spatial import Delaunay
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial import QhullError
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray


class PointCloudSurfaceSegmentation(Node):
    def __init__(self):
        super().__init__('pointcloud_surface_segmentation')

        self.declare_parameter('cloud_topic', '/octomap_point_cloud_centers')
        self.declare_parameter('marker_topic', '/segmented_surface_markers')
        self.declare_parameter('voxel_size', 0.10)
        self.declare_parameter('distance_threshold', 0.08)
        self.declare_parameter('horizontal_normal_threshold', 0.90)
        self.declare_parameter('min_plane_points', 20)
        self.declare_parameter('max_planes', 4)
        self.declare_parameter('component_cluster_tolerance', 0.22)
        self.declare_parameter('min_component_points', 8)
        self.declare_parameter('footprint_alpha', 0.25)
        self.declare_parameter('min_footprint_area', 0.02)
        self.declare_parameter('floor_clearance', 0.08)
        self.declare_parameter('floor_surface_height', float('nan'))
        self.declare_parameter('floor_reference_min_points', 100)
        self.declare_parameter('render_floor_outline', True)
        self.declare_parameter('plane_thickness', 0.10)
        self.declare_parameter('publish_residual', True)
        self.declare_parameter('candidate_edge_margin', 0.60)
        self.declare_parameter('candidate_min_robot_distance', 0.75)
        self.declare_parameter('explored_radius', 0.80)
        self.declare_parameter('trajectory_sample_distance', 0.20)
        self.declare_parameter('candidate_topic', '/red_surface_candidate_goal')

        cloud_topic = self.get_parameter('cloud_topic').value
        marker_topic = self.get_parameter('marker_topic').value
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.distance_threshold = float(self.get_parameter('distance_threshold').value)
        self.horizontal_normal_threshold = float(
            self.get_parameter('horizontal_normal_threshold').value)
        self.min_plane_points = int(self.get_parameter('min_plane_points').value)
        self.max_planes = int(self.get_parameter('max_planes').value)
        self.component_cluster_tolerance = float(
            self.get_parameter('component_cluster_tolerance').value)
        self.min_component_points = int(
            self.get_parameter('min_component_points').value)
        self.footprint_alpha = float(self.get_parameter('footprint_alpha').value)
        self.min_footprint_area = float(
            self.get_parameter('min_footprint_area').value)
        self.floor_clearance = float(self.get_parameter('floor_clearance').value)
        self.floor_surface_height = float(
            self.get_parameter('floor_surface_height').value)
        self.floor_reference_min_points = int(
            self.get_parameter('floor_reference_min_points').value)
        self.render_floor_outline = bool(
            self.get_parameter('render_floor_outline').value)
        self.plane_thickness = float(self.get_parameter('plane_thickness').value)
        self.publish_residual = bool(self.get_parameter('publish_residual').value)
        self.candidate_edge_margin = float(
            self.get_parameter('candidate_edge_margin').value)
        self.candidate_min_robot_distance = float(
            self.get_parameter('candidate_min_robot_distance').value)
        self.explored_radius = float(self.get_parameter('explored_radius').value)
        self.trajectory_sample_distance = float(
            self.get_parameter('trajectory_sample_distance').value)
        candidate_topic = self.get_parameter('candidate_topic').value

        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(
            PointCloud2, cloud_topic, self.cloud_callback, qos)
        self.marker_publisher = self.create_publisher(MarkerArray, marker_topic, 10)
        self.candidate_publisher = self.create_publisher(PointStamped, candidate_topic, 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.odometry_trajectory = []
        self.get_logger().info(
            f'Segmenting {cloud_topic} and publishing geometric surfaces on {marker_topic}')

    def cloud_callback(self, message):
        points = np.asarray([
            [float(point[0]), float(point[1]), float(point[2])]
            for point in pc2.read_points(
                message, field_names=('x', 'y', 'z'), skip_nans=True)
        ], dtype=np.float64)

        if len(points) < self.min_plane_points:
            return

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        downsampled = np.asarray(cloud.points)
        if len(downsampled) < self.min_plane_points:
            return

        planes, remaining = self._segment_horizontal_planes(cloud)
        robot_xy = self._robot_xy(message.header.frame_id)
        self._record_odometry_pose(robot_xy)
        plane_summary = [
            f"z={plane['height']:.2f}, points={len(plane['points'])}"
            for plane in planes]
        self.get_logger().info(
            f"surfaces={len(planes)} [{', '.join(plane_summary)}], "
            f"residual_points={len(remaining)}",
            throttle_duration_sec=5.0)
        self.marker_publisher.publish(
            self._make_markers(
                message.header.frame_id, planes, remaining, robot_xy,
                self.odometry_trajectory))

    def _robot_xy(self, frame_id):
        try:
            transform = self.tf_buffer.lookup_transform(
                frame_id, 'base_link', rclpy.time.Time())
            return np.array([
                transform.transform.translation.x,
                transform.transform.translation.y], dtype=np.float64)
        except Exception:
            return None

    def _record_odometry_pose(self, robot_xy):
        if robot_xy is None:
            return
        if (not self.odometry_trajectory
                or np.linalg.norm(robot_xy - self.odometry_trajectory[-1])
                >= self.trajectory_sample_distance):
            self.odometry_trajectory.append(np.copy(robot_xy))

    def _segment_horizontal_planes(self, cloud):
        remaining = cloud
        planes = []

        for _ in range(self.max_planes):
            if len(remaining.points) < self.min_plane_points:
                break

            model, inlier_indices = remaining.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=3,
                num_iterations=150)
            if len(inlier_indices) < self.min_plane_points:
                break

            normal = np.asarray(model[:3], dtype=np.float64)
            normal /= max(np.linalg.norm(normal), 1e-12)
            if abs(normal[2]) < self.horizontal_normal_threshold:
                mask = np.ones(len(remaining.points), dtype=bool)
                mask[inlier_indices] = False
                remaining = remaining.select_by_index(inlier_indices, invert=True)
                continue

            inlier_cloud = remaining.select_by_index(inlier_indices)
            inlier_points = np.asarray(inlier_cloud.points)
            components = self._split_xy_components(inlier_points)
            for component in components:
                planes.append({
                    'points': component,
                    'height': float(np.mean(component[:, 2])),
                    'normal': normal,
                })
            remaining = remaining.select_by_index(inlier_indices, invert=True)

        planes.sort(key=lambda plane: plane['height'])
        return planes, np.asarray(remaining.points)

    def _split_xy_components(self, points):
        """Split a plane into disconnected observed footprints in XY."""
        if len(points) < self.min_component_points:
            return []

        component_cloud = o3d.geometry.PointCloud()
        component_cloud.points = o3d.utility.Vector3dVector(points)
        labels = np.asarray(component_cloud.cluster_dbscan(
            eps=self.component_cluster_tolerance,
            min_points=1))
        components = []
        for label in range(labels.max() + 1):
            component = points[labels == label]
            if len(component) >= self.min_component_points:
                components.append(component)
        return components

    def _make_markers(self, frame_id, planes, residual, robot_xy,
                      odometry_trajectory):
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        floor_height = self._floor_reference_height(planes)
        candidate = self._select_red_candidate(
            planes, floor_height, robot_xy, odometry_trajectory)

        for index, plane in enumerate(planes):
            points = plane['points']
            is_floor = abs(plane['height'] - floor_height) <= self.floor_clearance
            triangles, boundary_edges = self._alpha_shape_xy(
                points[:, :2], self.footprint_alpha)
            footprint_area = sum(
                abs(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
                / 2.0 for triangle in triangles)
            if not triangles or footprint_area < self.min_footprint_area:
                continue

            if not is_floor:
                markers.markers.append(self._make_footprint_marker(
                    frame_id, stamp, index, triangles, points, plane['height']))
            if not is_floor or self.render_floor_outline:
                markers.markers.append(self._make_outline_marker(
                    frame_id, stamp, index + 1000, boundary_edges, points,
                    plane['height'], is_floor))

        if self.publish_residual and len(residual) > 0:
            residual_marker = Marker()
            residual_marker.header.frame_id = frame_id
            residual_marker.header.stamp = stamp
            residual_marker.ns = 'unsegmented_points'
            residual_marker.id = 0
            residual_marker.type = Marker.POINTS
            residual_marker.action = Marker.ADD
            residual_marker.scale.x = self.voxel_size
            residual_marker.scale.y = self.voxel_size
            residual_marker.color.r = 0.1
            residual_marker.color.g = 0.8
            residual_marker.color.b = 0.9
            residual_marker.color.a = 0.8
            residual_marker.points = [
                Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
                for point in residual
            ]
            markers.markers.append(residual_marker)

        candidate_marker = Marker()
        candidate_marker.header.frame_id = frame_id
        candidate_marker.header.stamp = stamp
        candidate_marker.ns = 'red_surface_candidate'
        candidate_marker.id = 0
        candidate_marker.action = Marker.ADD if candidate is not None else Marker.DELETE
        candidate_marker.type = Marker.SPHERE
        candidate_marker.scale.x = candidate_marker.scale.y = candidate_marker.scale.z = 0.35
        candidate_marker.color.r = 0.1
        candidate_marker.color.g = 1.0
        candidate_marker.color.b = 0.1
        candidate_marker.color.a = 1.0
        if candidate is not None:
            candidate_marker.pose.position.x = float(candidate[0])
            candidate_marker.pose.position.y = float(candidate[1])
            candidate_marker.pose.position.z = float(candidate[2])
            goal = PointStamped()
            goal.header.frame_id = frame_id
            goal.header.stamp = stamp
            goal.point.x = float(candidate[0])
            goal.point.y = float(candidate[1])
            goal.point.z = float(candidate[2])
            self.candidate_publisher.publish(goal)
        markers.markers.append(candidate_marker)

        path_marker = Marker()
        path_marker.header.frame_id = frame_id
        path_marker.header.stamp = stamp
        path_marker.ns = 'red_surface_candidate_path'
        path_marker.id = 1
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD if candidate is not None and robot_xy is not None else Marker.DELETE
        path_marker.scale.x = 0.04
        path_marker.color.r = 0.1
        path_marker.color.g = 1.0
        path_marker.color.b = 0.1
        path_marker.color.a = 0.9
        if candidate is not None and robot_xy is not None:
            path_marker.points = [
                Point(x=float(robot_xy[0]), y=float(robot_xy[1]), z=float(candidate[2])),
                Point(x=float(candidate[0]), y=float(candidate[1]), z=float(candidate[2])),
            ]
        markers.markers.append(path_marker)

        return markers

    def _floor_reference_height(self, planes):
        if math.isfinite(self.floor_surface_height):
            return self.floor_surface_height

        large_planes = [
            plane['height'] for plane in planes
            if len(plane['points']) >= self.floor_reference_min_points]
        if large_planes:
            return min(large_planes)
        return min((plane['height'] for plane in planes), default=float('inf'))

    def _select_red_candidate(self, planes, floor_height, robot_xy,
                              odometry_trajectory):
        """Select red interior space not already covered by robot travel."""
        candidates = []
        red = PointCloudSurfaceSegmentation._height_color(0.25)
        for plane in planes:
            if abs(plane['height'] - floor_height) <= self.floor_clearance:
                continue
            if PointCloudSurfaceSegmentation._height_color(plane['height']) != red:
                continue

            triangles, boundary_edges = self._alpha_shape_xy(
                plane['points'][:, :2], self.footprint_alpha)
            if not triangles or not boundary_edges:
                continue
            # Select only from rendered triangle centroids. Raw plane points can
            # be rejected by the alpha shape, which previously placed the goal
            # outside the visible red footprint.
            for triangle in triangles:
                point = np.mean(triangle, axis=0)
                point = np.array([point[0], point[1], plane['height']])
                clearance = min(
                    self._point_segment_distance(point[:2], edge[0], edge[1])
                    for edge in boundary_edges)
                distance = (np.linalg.norm(point[:2] - robot_xy)
                            if robot_xy is not None else float('inf'))
                if (robot_xy is not None
                        and distance < self.candidate_min_robot_distance):
                    continue
                if odometry_trajectory:
                    explored_distance = min(
                        np.linalg.norm(point[:2] - pose)
                        for pose in odometry_trajectory)
                    if explored_distance < self.explored_radius:
                        continue
                else:
                    explored_distance = float('inf')
                if clearance >= self.candidate_edge_margin:
                    candidates.append((clearance, explored_distance, -distance, point))

        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]

    @staticmethod
    def _point_segment_distance(point, start, end):
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared == 0.0:
            return float(np.linalg.norm(point - start))
        fraction = np.clip(
            np.dot(point - start, segment) / length_squared, 0.0, 1.0)
        projection = start + fraction * segment
        return float(np.linalg.norm(point - projection))

    def _make_footprint_marker(self, frame_id, stamp, marker_id, triangles,
                               points, height):
        z = float(np.mean(points[:, 2]))
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'surface_footprints'
        marker.id = marker_id
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        marker.color.r, marker.color.g, marker.color.b = PointCloudSurfaceSegmentation._height_color(height)
        marker.color.a = 0.65
        marker.points = [
            Point(x=float(vertex[0]), y=float(vertex[1]), z=z)
            for triangle in triangles for vertex in triangle]
        return marker

    def _make_outline_marker(self, frame_id, stamp, marker_id, boundary_edges, points,
                             height, is_floor):
        z = float(np.mean(points[:, 2]))

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'surface_outlines'
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        if is_floor:
            marker.color.r = marker.color.g = marker.color.b = 0.35
        else:
            marker.color.r, marker.color.g, marker.color.b = PointCloudSurfaceSegmentation._height_color(height)
        marker.color.a = 0.25 if is_floor else 1.0
        marker.points = [
            Point(x=float(vertex[0]), y=float(vertex[1]), z=z)
            for edge in boundary_edges for vertex in edge]
        return marker

    @staticmethod
    def _height_color(height):
        """Use a stable height palette so stacked surfaces are easy to distinguish."""
        palette = (
            (0.95, 0.25, 0.15),
            (0.95, 0.70, 0.10),
            (0.20, 0.75, 0.90),
            (0.35, 0.85, 0.35),
            (0.75, 0.35, 0.90),
        )
        return palette[int(round(height / 0.05)) % len(palette)]

    @staticmethod
    def _alpha_shape_xy(points, alpha):
        """Return triangles and boundary edges without filling observed gaps."""
        unique = np.unique(points, axis=0)
        if len(unique) < 4:
            return [], []

        # A scan line has no 2D footprint. It may be a wall or a partial
        # return, but it cannot be triangulated into a filled surface.
        centered = unique - np.mean(unique, axis=0)
        if np.linalg.matrix_rank(centered) < 2:
            return [], []

        try:
            triangulation = Delaunay(unique)
        except QhullError:
            return [], []
        kept_triangles = []
        edge_counts = {}
        for simplex in triangulation.simplices:
            triangle = unique[simplex]
            side_a = np.linalg.norm(triangle[1] - triangle[0])
            side_b = np.linalg.norm(triangle[2] - triangle[1])
            side_c = np.linalg.norm(triangle[0] - triangle[2])
            semiperimeter = (side_a + side_b + side_c) / 2.0
            area = max(np.sqrt(max(
                semiperimeter * (semiperimeter - side_a)
                * (semiperimeter - side_b) * (semiperimeter - side_c), 0.0)),
                1e-12)
            circumradius = side_a * side_b * side_c / (4.0 * area)
            if circumradius > alpha:
                continue

            kept_triangles.append(triangle)
            for first, second in ((simplex[0], simplex[1]),
                                  (simplex[1], simplex[2]),
                                  (simplex[2], simplex[0])):
                edge = tuple(sorted((int(first), int(second))))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1

        boundary_edges = [
            (unique[first], unique[second])
            for (first, second), count in edge_counts.items()
            if count == 1]
        return kept_triangles, boundary_edges


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSurfaceSegmentation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
