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
    'pallet_surface_z': None,   # Z height of the pallet top in the map frame
    'pallet_height_tolerance': 0.12,  # Allowed frontier height variation on the pallet
    'pallet_bounds': None,      # (min_x, max_x, min_y, max_y) in the map frame
    'pallet_edge_margin': 0.5,  # Do not navigate this close to a pallet edge
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
        self.last_stats = {
            'point_count': 0,
            'boundary_count': 0,
            'raw_goal_count': 0,
            'filtered_goal_count': 0,
            'raw_goal_z_range': None,
            'rejections': {},
        }

    def process_pointcloud(self, pts_array, robot_pose, blacklist=None, visited=None):
        """
        Main pipeline to process a raw point cloud into a navigation goal.
        """
        if blacklist is None: blacklist = []
        if visited is None: visited = []

        # 1. Extract raw boundary points, cluster them into goals, and get their point counts (sizes)
        boundary_pts, raw_goals, raw_sizes = self.extract_3d_boundary_edges(pts_array)
        self.last_stats['point_count'] = len(pts_array)
        self.last_stats['boundary_count'] = 0 if boundary_pts is None else len(boundary_pts)
        self.last_stats['raw_goal_count'] = 0 if raw_goals is None else len(raw_goals)
        self.last_stats['filtered_goal_count'] = 0
        self.last_stats['raw_goal_z_range'] = None
        self.last_stats['rejections'] = {}

        if raw_goals is not None and len(raw_goals) > 0:
            self.last_stats['raw_goal_z_range'] = (
                float(np.min(raw_goals[:, 2])),
                float(np.max(raw_goals[:, 2])))
        
        # If no boundaries are found, return early with an empty marker array
        if boundary_pts is None or len(boundary_pts) == 0:
            return None, MarkerArray()

        # 2. Filter out goals that are unreachable, already visited, or blacklisted
        filtered_goals, filtered_sizes = self.filter_goals(
            raw_goals, raw_sizes, robot_pose, blacklist, visited
        )
        self.last_stats['filtered_goal_count'] = len(filtered_goals)

        # 3. Select the single best goal for navigation based on the chosen strategy
        best_goal = self._select_best_goal(filtered_goals, filtered_sizes, robot_pose)

        # 4. Generate visual markers for RViz to debug and monitor the pipeline
        markers = self._generate_markers(boundary_pts, raw_goals, filtered_goals, best_goal)

        return best_goal, markers

    def process_occupancy_grid(self, grid, robot_pose, blacklist=None, visited=None):
        """Find free-to-unknown transitions in the 2D map as exploration frontiers."""
        if blacklist is None:
            blacklist = []
        if visited is None:
            visited = []

        width = int(grid.info.width)
        height = int(grid.info.height)
        data = np.asarray(grid.data, dtype=np.int16).reshape((height, width))
        free = (data >= 0) & (data < 50)
        unknown = data < 0
        frontier_mask = np.zeros((height, width), dtype=bool)

        frontier_mask[1:, :] |= free[1:, :] & unknown[:-1, :]
        frontier_mask[:-1, :] |= free[:-1, :] & unknown[1:, :]
        frontier_mask[:, 1:] |= free[:, 1:] & unknown[:, :-1]
        frontier_mask[:, :-1] |= free[:, :-1] & unknown[:, 1:]

        y_cells, x_cells = np.nonzero(frontier_mask)
        resolution = float(grid.info.resolution)
        origin_x = float(grid.info.origin.position.x)
        origin_y = float(grid.info.origin.position.y)
        frontier_z = self.params.get('pallet_surface_z')
        if frontier_z is None:
            frontier_z = robot_pose[2]
        boundary_pts = np.column_stack((
            origin_x + (x_cells + 0.5) * resolution,
            origin_y + (y_cells + 0.5) * resolution,
            np.full(len(x_cells), frontier_z)))

        self.last_stats['point_count'] = width * height
        self.last_stats['boundary_count'] = len(boundary_pts)
        self.last_stats['raw_goal_count'] = 0
        self.last_stats['filtered_goal_count'] = 0
        self.last_stats['raw_goal_z_range'] = None
        self.last_stats['rejections'] = {}

        if len(boundary_pts) == 0:
            return None, MarkerArray()

        frontier_pcd = o3d.geometry.PointCloud()
        frontier_pcd.points = o3d.utility.Vector3dVector(boundary_pts)
        labels = np.asarray(frontier_pcd.cluster_dbscan(
            eps=max(2.0 * resolution, self.params['neighbor_radius']), min_points=1))

        raw_goals = []
        raw_sizes = []
        for label in range(labels.max() + 1):
            cluster_pts = boundary_pts[labels == label]
            if len(cluster_pts) < self.params['min_neighbors']:
                continue
            center = np.mean(cluster_pts, axis=0)
            raw_goals.append(cluster_pts[np.argmin(np.linalg.norm(cluster_pts - center, axis=1))])
            raw_sizes.append(len(cluster_pts))

        raw_goals = np.asarray(raw_goals)
        raw_sizes = np.asarray(raw_sizes)
        self.last_stats['raw_goal_count'] = len(raw_goals)
        if len(raw_goals) > 0:
            self.last_stats['raw_goal_z_range'] = (
                float(np.min(raw_goals[:, 2])), float(np.max(raw_goals[:, 2])))

        filtered_goals, filtered_sizes = self.filter_goals(
            raw_goals, raw_sizes, robot_pose, blacklist, visited)
        self.last_stats['filtered_goal_count'] = len(filtered_goals)
        best_goal = self._select_best_goal(filtered_goals, filtered_sizes, robot_pose)
        return best_goal, self._generate_markers(
            boundary_pts, raw_goals, filtered_goals, best_goal)

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
        pallet_surface_z = self.params.get('pallet_surface_z')
        pallet_height_tolerance = self.params['pallet_height_tolerance']
        pallet_bounds = self.params.get('pallet_bounds')
        pallet_edge_margin = self.params['pallet_edge_margin']
        
        # Pre-compute squared radii to avoid computing square roots in the loop (optimization)
        visited_rad_sq = self.params['visited_radius'] ** 2
        blacklist_rad_sq = self.params['blacklist_radius'] ** 2
        rejections = self.last_stats['rejections']

        def reject(reason):
            rejections[reason] = rejections.get(reason, 0) + 1

        for idx, g in enumerate(raw_goals):
            # 1. Height check: Don't target floors above/below current reach
            if abs(g[2] - robot_pose[2]) > max_z_diff:
                reject('robot_height')
                continue

            # In a floor-is-lava scenario, only the pallet surface is traversable.
            # This rejects floor frontiers before they can be selected as goals.
            if pallet_surface_z is not None and abs(g[2] - pallet_surface_z) > pallet_height_tolerance:
                reject('pallet_height')
                continue

            # The cloud boundary includes the physical pallet perimeter. Only
            # accept candidates with a margin from each configured pallet edge.
            if pallet_bounds is not None:
                min_x, max_x, min_y, max_y = pallet_bounds
                inside_pallet = (
                    min_x + pallet_edge_margin <= g[0] <= max_x - pallet_edge_margin
                    and min_y + pallet_edge_margin <= g[1] <= max_y - pallet_edge_margin)
                if not inside_pallet:
                    reject('pallet_edge')
                    continue
            
            # 2. Min distance check: Ignore goals directly under the robot
            if np.linalg.norm(g - robot_pose) < min_dist:
                reject('too_close')
                continue
                
            is_invalid = False
            
            # 3. Visited check: Ignore goals near areas we've already explored
            for v in visited:
                if np.sum((g - v)**2) < visited_rad_sq:
                    is_invalid = True
                    reject('visited')
                    break
            if is_invalid: continue
            
            # 4. Blacklist check: Ignore goals previously marked as unreachable
            for b in blacklist:
                if np.sum((g - b)**2) < blacklist_rad_sq:
                    is_invalid = True
                    reject('blacklisted')
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