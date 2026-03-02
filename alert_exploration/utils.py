import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from rclpy.time import Time

DEFAULT_PARAMS = {
    # --- Perception Settings ---
    'voxel_size': 0.15,
    'neighbor_radius': 0.45,
    'min_neighbors': 5,
    
    # --- Filtering & Memory Settings ---
    'min_dist_from_robot': 0.8,
    'max_height_diff': 1.5,
    'visited_radius': 1.5,
    'blacklist_radius': 1.0,
    
    # --- Framework Settings ---
    'map_frame': 'map',
    'goal_strategy': 'closest' # Can be 'closest' or 'random'
}

class PointCloudFrontierDetector:
    def __init__(self, params=None):
        self.params = DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

    def process_pointcloud(self, pts_array, robot_pose, blacklist=None, visited=None):
        if blacklist is None: blacklist = []
        if visited is None: visited = []

        # 1. Extract raw boundaries, raw goals, AND their sizes
        boundary_pts, raw_goals, raw_sizes = self.extract_3d_boundary_edges(pts_array)
        
        if boundary_pts is None or len(boundary_pts) == 0:
            return None, MarkerArray()

        # 2. Filter goals (and keep sizes aligned)
        filtered_goals, filtered_sizes = self.filter_goals(raw_goals, raw_sizes, robot_pose, blacklist, visited)

        # 3. Select the single best goal for navigation using the new math
        best_goal = self._select_best_goal(filtered_goals, filtered_sizes, robot_pose)

        # 4. Generate all markers for RViz visualization
        markers = self._generate_markers(boundary_pts, raw_goals, filtered_goals, best_goal)

        return best_goal, markers

    def extract_3d_boundary_edges(self, pts_array):
        """Finds frontier edges and clusters them via DBSCAN."""
        voxel_size = self.params['voxel_size']
        neighbor_radius = self.params['neighbor_radius']
        min_neighbors = self.params['min_neighbors']

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_array)
        pcd = pcd.voxel_down_sample(voxel_size)
        
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=neighbor_radius * 1.5, max_nn=30)
        )
        pcd.orient_normals_to_align_with_direction(np.array([0., 0., 1.]))

        pts = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        if len(pts) == 0: return None, None

        tree = KDTree(pts)
        neighbors_list = tree.query_ball_point(pts, r=neighbor_radius, workers=-1)
        is_boundary_node = np.zeros(len(pts), dtype=bool)
        
        # Edge Detection
        for i in range(len(pts)):
            if normals[i][2] < 0.8: continue ## 0.7 radians from vertical; can tranverse ramps but not walls
            neigh_indices = neighbors_list[i]
            if len(neigh_indices) < min_neighbors: continue
                
            neigh_pts = pts[neigh_indices]
            neigh_centroid = np.mean(neigh_pts, axis=0)
            offset_dist = np.linalg.norm(neigh_centroid - pts[i])
            
            if offset_dist > (0.35 * neighbor_radius):
                is_boundary_node[i] = True
                
        boundary_pts = pts[is_boundary_node]
        if len(boundary_pts) == 0: return None, None

        # Clustering
        boundary_pcd = o3d.geometry.PointCloud()
        boundary_pcd.points = o3d.utility.Vector3dVector(boundary_pts)
        labels = np.array(boundary_pcd.cluster_dbscan(eps=neighbor_radius * 1.5, min_points=min_neighbors))
        
        goal_centroids = []
        cluster_sizes = [] 
        
        if len(labels) > 0:
            for i in range(labels.max() + 1):
                cluster_pts = boundary_pts[labels == i]
                if np.linalg.norm(np.max(cluster_pts, axis=0) - np.min(cluster_pts, axis=0)) < 0.3: continue
                
                geometric_center = np.mean(cluster_pts, axis=0)
                dists = np.linalg.norm(cluster_pts - geometric_center, axis=1)
                
                goal_centroids.append(cluster_pts[np.argmin(dists)]) # Medoid
                cluster_sizes.append(len(cluster_pts)) # --- NEW: Save point count ---

        return boundary_pts, np.array(goal_centroids), np.array(cluster_sizes)

    def filter_goals(self, raw_goals, raw_sizes, robot_pose, blacklist, visited):
        if raw_goals is None or len(raw_goals) == 0:
            return [], []
            
        valid_goals = []
        valid_sizes = [] # Track sizes of goals that pass the filter
        
        min_dist = self.params['min_dist_from_robot']
        max_z_diff = self.params['max_height_diff']
        visited_rad_sq = self.params['visited_radius'] ** 2
        blacklist_rad_sq = self.params['blacklist_radius'] ** 2

        # Iterate using enumerate to get the index for the corresponding size
        for idx, g in enumerate(raw_goals):
            if abs(g[2] - robot_pose[2]) > max_z_diff: continue
            if np.linalg.norm(g - robot_pose) < min_dist: continue
                
            is_invalid = False
            for v in visited:
                if np.sum((g - v)**2) < visited_rad_sq:
                    is_invalid = True; break
            if is_invalid: continue
            
            for b in blacklist:
                if np.sum((g - b)**2) < blacklist_rad_sq:
                    is_invalid = True; break
            if is_invalid: continue
            
            valid_goals.append(g)
            valid_sizes.append(raw_sizes[idx]) # Keep the size aligned
            
        return np.array(valid_goals), np.array(valid_sizes)

    def _select_best_goal(self, valid_goals, valid_sizes, robot_pose):
        """Selects the best target using a Utility function (Size vs Distance)."""
        if valid_goals is None or len(valid_goals) == 0:
            return None
            
        strategy = self.params.get('goal_strategy', 'utility')
        
        if strategy == 'closest':
            dists = np.linalg.norm(valid_goals - robot_pose, axis=1)
            return valid_goals[np.argmin(dists)]
            
        elif strategy == 'utility':
            # --- UTILITY SCORING ---
            dists = np.linalg.norm(valid_goals - robot_pose, axis=1)
            
            max_dist = np.max(dists) if np.max(dists) > 0 else 1.0
            norm_dists = dists / max_dist
            
            max_size = np.max(valid_sizes) if np.max(valid_sizes) > 0 else 1.0
            norm_sizes = valid_sizes / max_size
            
            # Tuning Weights: Increase distance_weight to stay local, 
            # increase size_weight to jump to huge open areas.
            size_weight = 1.5
            distance_weight = 1.0
            
            # Calculate utility
            utilities = (size_weight * norm_sizes) - (distance_weight * norm_dists)
            
            # Return the goal with the highest utility score
            best_idx = np.argmax(utilities)
            return valid_goals[best_idx]

    def _generate_markers(self, boundary_pts, raw_goals, filtered_goals, best_goal):
        """Packages arrays into a ROS 2 MarkerArray for RViz."""
        ma = MarkerArray()
        stamp = Time().to_msg()
        map_frame = self.params['map_frame']
        
        # Helper to create standard Marker properties
        def create_marker(mid, ns, color, scale, m_type=Marker.SPHERE_LIST):
            m = Marker()
            m.header.frame_id = map_frame
            m.header.stamp = stamp
            m.ns = ns
            m.id = mid
            m.type = m_type
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color.r, m.color.g, m.color.b, m.color.a = color
            return m

        # 1. Edges (Cyan)
        if boundary_pts is not None and len(boundary_pts) > 0:
            m_edges = create_marker(0, "edges", (0.0, 1.0, 1.0, 0.5), self.params['voxel_size'])
            for pt in boundary_pts:
                p = Point(); p.x, p.y, p.z = float(pt[0]), float(pt[1]), float(pt[2])
                m_edges.points.append(p)
            ma.markers.append(m_edges)

        # 2. Raw Rejected Goals (Red)
        if raw_goals is not None and len(raw_goals) > 0:
            m_raw = create_marker(1, "raw_goals", (1.0, 0.0, 0.0, 0.4), 0.2)
            for pt in raw_goals:
                p = Point(); p.x, p.y, p.z = float(pt[0]), float(pt[1]), float(pt[2])
                m_raw.points.append(p)
            ma.markers.append(m_raw)

        # 3. Filtered Candidate Goals (Yellow)
        if filtered_goals is not None and len(filtered_goals) > 0:
            m_goals = create_marker(2, "filtered_goals", (1.0, 1.0, 0.0, 0.7), 0.3)
            for pt in filtered_goals:
                p = Point(); p.x, p.y, p.z = float(pt[0]), float(pt[1]), float(pt[2])
                m_goals.points.append(p)
            ma.markers.append(m_goals)

        # 4. Best Goal (Large Green)
        if best_goal is not None:
            m_best = create_marker(3, "best_goal", (0.0, 1.0, 0.0, 1.0), 0.5, m_type=Marker.SPHERE)
            m_best.pose.position.x = float(best_goal[0])
            m_best.pose.position.y = float(best_goal[1])
            m_best.pose.position.z = float(best_goal[2])
            ma.markers.append(m_best)

        return ma