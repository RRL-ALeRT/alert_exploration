import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from rclpy.time import Time

# Default configuration parameters for the frontier detection pipeline
DEFAULT_PARAMS = {
    # --- Perception Settings ---
    'voxel_size': 0.1,         # Downsampling grid size (meters)
    'neighbor_radius': 0.5,    # Radius for local neighborhood search
    'min_neighbors': 5,         # Minimum points required in a radius to be considered valid
    
    # --- Filtering & Memory Settings ---
    'min_dist_from_robot': 0.8, # Ignore goals too close to the robot (meters)
    'max_height_diff': 1.5,     # Ignore goals too far above/below the robot (meters)
    'visited_radius': 1.5,      # Radius around past goals to mark as explored
    'blacklist_radius': 1.0,    # Radius around unreachable/failed goals
    
    # --- Framework Settings ---
    'map_frame': 'map',         # ROS coordinate frame for visualization
    'goal_strategy': 'utility'  # Strategy to pick the best goal ('closest' or 'utility')
}

class PointCloudFrontierDetector:
    """
    Detects exploration frontiers from 3D point clouds.
    It identifies boundary edges in the point cloud, clusters them into goals, 
    filters out invalid/visited ones, and selects the best navigation target.
    """
    
    def __init__(self, params=None):
        self.params = DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

    def process_pointcloud(self, pts_array, robot_pose, blacklist=None, visited=None):
        """
        Main pipeline to process a raw point cloud into a navigation goal.
        """
        if blacklist is None: blacklist = []
        if visited is None: visited = []

        # 1. Extract raw boundary points, cluster them into goals, and get their point counts (sizes)
        boundary_pts, raw_goals, raw_sizes = self.extract_3d_boundary_edges(pts_array)
        
        # If no boundaries are found, return early with an empty marker array
        if boundary_pts is None or len(boundary_pts) == 0:
            return None, MarkerArray()

        # 2. Filter out goals that are unreachable, already visited, or blacklisted
        filtered_goals, filtered_sizes = self.filter_goals(
            raw_goals, raw_sizes, robot_pose, blacklist, visited
        )

        # 3. Select the single best goal for navigation based on the chosen strategy
        best_goal = self._select_best_goal(filtered_goals, filtered_sizes, robot_pose)

        # 4. Generate visual markers for RViz to debug and monitor the pipeline
        markers = self._generate_markers(boundary_pts, raw_goals, filtered_goals, best_goal)

        return best_goal, markers

    def extract_3d_boundary_edges(self, pts_array):
        """
        Finds frontier edges by analyzing local point distributions and clusters them via DBSCAN.
        """
        voxel_size = self.params['voxel_size']
        neighbor_radius = self.params['neighbor_radius']
        min_neighbors = self.params['min_neighbors']

        # Convert numpy array to Open3D point cloud format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_array)
        
        # Downsample for computational efficiency
        pcd = pcd.voxel_down_sample(voxel_size)
        
        # Estimate surface normals. The radius is slightly larger than the neighbor search 
        # to ensure smooth normal estimation.
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=neighbor_radius * 1.5, max_nn=30)
        )
        # Orient normals upwards (towards +Z) to standardize the z-component check later
        pcd.orient_normals_to_align_with_direction(np.array([0., 0., 1.]))

        pts = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        if len(pts) == 0: return None, None, None

        # Build a KDTree for fast spatial radius queries
        tree = KDTree(pts)
        neighbors_list = tree.query_ball_point(pts, r=neighbor_radius, workers=-1)
        
        # Boolean mask to track which points are boundary/frontier points
        is_boundary_node = np.zeros(len(pts), dtype=bool)
        
        # --- Edge Detection Logic ---
        for i in range(len(pts)):
            # Filter out steep walls: normals[i][2] is the Z component of the normal.
            # 0.8 corresponds to ~36 degrees from vertical. 
            # This allows the robot to traverse ramps but ignores sheer walls.
            if normals[i][2] < 0.8: 
                continue 
            
            neigh_indices = neighbors_list[i]
            if len(neigh_indices) < min_neighbors: 
                continue
                
            # A point is a boundary if its local neighborhood is highly asymmetrical.
            # We determine this by calculating the distance from the point to the centroid
            # of its neighbors.
            neigh_pts = pts[neigh_indices]
            neigh_centroid = np.mean(neigh_pts, axis=0)
            offset_dist = np.linalg.norm(neigh_centroid - pts[i])
            
            # If the offset is greater than 35% of the search radius, it's on an edge
            if offset_dist > (0.35 * neighbor_radius):
                is_boundary_node[i] = True
                
        boundary_pts = pts[is_boundary_node]
        if len(boundary_pts) == 0: return None, None, None

        # --- Clustering Logic ---
        # Group adjacent boundary points into distinct "frontier" clusters using DBSCAN
        boundary_pcd = o3d.geometry.PointCloud()
        boundary_pcd.points = o3d.utility.Vector3dVector(boundary_pts)
        labels = np.array(boundary_pcd.cluster_dbscan(eps=neighbor_radius * 1.5, min_points=min_neighbors))
        
        goal_centroids = []
        cluster_sizes = [] 
        
        if len(labels) > 0:
            for i in range(labels.max() + 1):
                cluster_pts = boundary_pts[labels == i]
                
                # Reject clusters that are too physically small (less than 30cm across)
                # This prevents the robot from targeting tiny artifacts or noise.
                bounding_box_size = np.linalg.norm(np.max(cluster_pts, axis=0) - np.min(cluster_pts, axis=0))
                if bounding_box_size < 0.3: 
                    continue
                
                # Find the 'medoid' of the cluster (the actual point closest to the geometric center)
                # This ensures the goal is an actual traversable coordinate, not a void in space.
                geometric_center = np.mean(cluster_pts, axis=0)
                dists = np.linalg.norm(cluster_pts - geometric_center, axis=1)
                
                goal_centroids.append(cluster_pts[np.argmin(dists)]) 
                cluster_sizes.append(len(cluster_pts))

        return boundary_pts, np.array(goal_centroids), np.array(cluster_sizes)

    def filter_goals(self, raw_goals, raw_sizes, robot_pose, blacklist, visited):
        """
        Removes goals based on distance, height difference, and memory (visited/blacklisted).
        """
        if raw_goals is None or len(raw_goals) == 0:
            return [], []
            
        valid_goals = []
        valid_sizes = [] 
        
        min_dist = self.params['min_dist_from_robot']
        max_z_diff = self.params['max_height_diff']
        
        # Pre-compute squared radii to avoid computing square roots in the loop (optimization)
        visited_rad_sq = self.params['visited_radius'] ** 2
        blacklist_rad_sq = self.params['blacklist_radius'] ** 2

        for idx, g in enumerate(raw_goals):
            # 1. Height check: Don't target floors above/below current reach
            if abs(g[2] - robot_pose[2]) > max_z_diff: continue
            
            # 2. Min distance check: Ignore goals directly under the robot
            if np.linalg.norm(g - robot_pose) < min_dist: continue
                
            is_invalid = False
            
            # 3. Visited check: Ignore goals near areas we've already explored
            for v in visited:
                if np.sum((g - v)**2) < visited_rad_sq:
                    is_invalid = True
                    break
            if is_invalid: continue
            
            # 4. Blacklist check: Ignore goals previously marked as unreachable
            for b in blacklist:
                if np.sum((g - b)**2) < blacklist_rad_sq:
                    is_invalid = True
                    break
            if is_invalid: continue
            
            # If all checks pass, keep the goal and its corresponding size
            valid_goals.append(g)
            valid_sizes.append(raw_sizes[idx]) 
            
        return np.array(valid_goals), np.array(valid_sizes)

    def _select_best_goal(self, valid_goals, valid_sizes, robot_pose):
        """
        Evaluates valid goals and returns the optimal one based on distance and cluster size.
        """
        if valid_goals is None or len(valid_goals) == 0:
            return None
            
        strategy = self.params.get('goal_strategy', 'utility')
        
        if strategy == 'closest':
            # Simple greedy approach: go to the nearest frontier
            dists = np.linalg.norm(valid_goals - robot_pose, axis=1)
            return valid_goals[np.argmin(dists)]
            
        elif strategy == 'utility':
            # --- UTILITY SCORING ---
            # Balances exploration efficiency by weighing the size of the frontier 
            # against the travel cost (distance).
            dists = np.linalg.norm(valid_goals - robot_pose, axis=1)
            
            # Normalize distances (0 to 1) so weights scale predictably
            max_dist = np.max(dists) if np.max(dists) > 0 else 1.0
            norm_dists = dists / max_dist
            
            # Normalize sizes (0 to 1)
            max_size = np.max(valid_sizes) if np.max(valid_sizes) > 0 else 1.0
            norm_sizes = valid_sizes / max_size
            
            # Tuning Weights: 
            # size_weight > distance_weight encourages jumping to larger unexplored areas.
            # distance_weight > size_weight encourages methodical, local clearing.
            size_weight = 1.5
            distance_weight = 1.0
            
            # Higher utility is better (bigger size adds points, further distance subtracts points)
            utilities = (size_weight * norm_sizes) - (distance_weight * norm_dists)
            
            best_idx = np.argmax(utilities)
            return valid_goals[best_idx]

    def _generate_markers(self, boundary_pts, raw_goals, filtered_goals, best_goal):
        """
        Converts numpy arrays into a ROS 2 MarkerArray for RViz visualization.
        Provides distinct colors/sizes for different stages of the pipeline.
        """
        ma = MarkerArray()
        stamp = Time().to_msg()
        map_frame = self.params['map_frame']
        
        # Helper function to reduce boilerplate when creating markers
        def create_marker(mid, ns, color, scale, m_type=Marker.SPHERE_LIST):
            m = Marker()
            m.header.frame_id = map_frame
            m.header.stamp = stamp
            m.ns = ns
            m.id = mid
            m.type = m_type
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = scale
            # Color is an RGBA tuple
            m.color.r, m.color.g, m.color.b, m.color.a = color
            return m

        # 1. Detected Edges (Cyan, transparent)
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

        # 4. Final Best Goal (Large solid Green sphere)
        if best_goal is not None:
            m_best = create_marker(3, "best_goal", (0.0, 1.0, 0.0, 1.0), 0.5, m_type=Marker.SPHERE)
            m_best.pose.position.x = float(best_goal[0])
            m_best.pose.position.y = float(best_goal[1])
            m_best.pose.position.z = float(best_goal[2])
            ma.markers.append(m_best)

        return ma