import heapq
import math
import scipy.interpolate as si
import numpy as np

from nav_msgs.msg import OccupancyGrid


def add_unexplored_edges(map_msg, edge_size):
    width = map_msg.info.width
    height = map_msg.info.height
    origin_x = map_msg.info.origin.position.x
    origin_y = map_msg.info.origin.position.y
    resolution = map_msg.info.resolution

    data = np.array(map_msg.data).reshape((height, width))

    # Create an extended map with increased edges
    new_height = height + 2 * edge_size
    new_width = width + 2 * edge_size
    new_origin_x = origin_x - edge_size * resolution
    new_origin_y = origin_y - edge_size * resolution

    extended_map = np.ones((new_height, new_width)) * -1  # Initialize with -1 for unexplored

    # Copy the original map data into the center of the new map
    extended_map[edge_size:edge_size + height, edge_size:edge_size + width] = data

    # Prepare the modified OccupancyGrid message
    modified_msg = OccupancyGrid()
    modified_msg.header = map_msg.header
    modified_msg.info = map_msg.info

    # Update the map info to reflect the new dimensions and origin
    modified_msg.info.width = new_width
    modified_msg.info.height = new_height
    modified_msg.info.origin.position.x = new_origin_x
    modified_msg.info.origin.position.y = new_origin_y

    # Flatten the extended map and assign it to the message
    modified_msg.data = extended_map.flatten().astype(np.int8).tolist()

    return modified_msg


def pure_pursuit(current_x, current_y, current_heading, path, index, speed, lookahead_distance, forward=True):
    closest_point = None
    if forward:
        v = speed  # Set the speed to a negative value to make the robot go in reverse
    else:
        v = -speed  # Set the speed to a negative value to make the robot go in reverse
    for i in range(index, len(path)):
        x = path[i][0]
        y = path[i][1]
        distance = math.hypot(current_x - x, current_y - y)
        if lookahead_distance < distance:
            closest_point = (x, y)
            index = i
            break
    if closest_point is not None:
        if forward:
            target_heading = math.atan2(closest_point[1] - current_y, closest_point[0] - current_x)
        else:
            target_heading = math.atan2(current_y - closest_point[1], current_x - closest_point[0])  # Reverse the atan2 arguments
        desired_steering_angle = target_heading - current_heading
    else:
        if forward:
            target_heading = math.atan2(path[-1][1] - current_y, path[-1][0] - current_x)
        else:
            target_heading = math.atan2(current_y - path[-1][1], current_x - path[-1][0])  # Reverse the atan2 arguments
        desired_steering_angle = target_heading - current_heading
        index = len(path) - 1
    if desired_steering_angle > math.pi:
        desired_steering_angle -= 2 * math.pi
    elif desired_steering_angle < -math.pi:
        desired_steering_angle += 2 * math.pi
    if desired_steering_angle > math.pi / 6 or desired_steering_angle < -math.pi / 6:
        sign = 1 if desired_steering_angle > 0 else -1
        desired_steering_angle = (sign * math.pi / 4)
        v = 0.0
    return v, desired_steering_angle, index


def pathLength(path):
    for i in range(len(path)):
        path[i] = (path[i][0],path[i][1])
        points = np.array(path)
    differences = np.diff(points, axis=0)
    distances = np.hypot(differences[:,0], differences[:,1])
    total_distance = np.sum(distances)
    return total_distance


def bspline_planning(array, sn):
    try:
        array = np.array(array)
        x = array[:, 0]
        y = array[:, 1]
        N = 2
        t = np.arange(len(x))
        x_tup = si.splrep(t, x, k=N)
        y_tup = si.splrep(t, y, k=N)

        x_list = list(x_tup)
        x_list[1] = np.pad(x_list[1], (0, 4), mode='constant')

        y_list = list(y_tup)
        y_list[1] = np.pad(y_list[1], (0, 4), mode='constant')

        ipl_t = np.linspace(0.0, len(x) - 1, sn)
        rx = si.splev(ipl_t, x_list)
        ry = si.splev(ipl_t, y_list)

        path = list(zip(rx, ry))
    except (TypeError, ValueError):
        path = array.tolist()

    return path


def world_to_map_coords(map_msg, world_x, world_y):
    origin_x = map_msg.info.origin.position.x
    origin_y = map_msg.info.origin.position.y
    resolution = map_msg.info.resolution

    map_x = int((world_x - origin_x) / resolution)
    map_y = int((world_y - origin_y) / resolution)
    return map_x, map_y


def map_to_world_coords(map_msg, map_x, map_y):
    origin_x = map_msg.info.origin.position.x
    origin_y = map_msg.info.origin.position.y
    resolution = map_msg.info.resolution

    world_x = map_x * resolution + origin_x
    world_y = map_y * resolution + origin_y
    return world_x, world_y


def yaw_from_quaternion(x, y, z, w):
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


def costmap(map_data: OccupancyGrid, expansion_size) -> OccupancyGrid:
    width = map_data.info.width
    height = map_data.info.height
    data = np.array(map_data.data).reshape((height, width))

    # Find walls (occupied cells)
    walls = np.where(data == 100)

    for i in range(-expansion_size,expansion_size+1):
        for j in range(-expansion_size,expansion_size+1):
            if i  == 0 and j == 0:
                continue
            x = walls[0]+i
            y = walls[1]+j
            x = np.clip(x,0,height-1)
            y = np.clip(y,0,width-1)
            data[x,y] = 100

    # Convert back to int8 array
    flattened_data = data.flatten().astype(np.int8)

    # Update the map_data with the expanded map
    map_data.data = flattened_data.tolist()  # Convert numpy array to list of int8
    return map_data


def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def astar(array, start, goal):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    open_set = {start}

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            data.reverse()
            return data, True

        close_set.add(current)

        if current in open_set:
            open_set.remove(current)

        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j

            if neighbor[1] < 0 or neighbor[1] >= array.shape[0] or neighbor[0] < 0 or neighbor[0] >= array.shape[1]:
                continue

            if array[neighbor[1]][neighbor[0]] == 100:
                continue

            if array[neighbor[1]][neighbor[0]] == -1:
                continue

            tentative_g_score = gscore[current] + heuristic(current, neighbor)

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in open_set:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
                open_set.add(neighbor)

    return [], False


def get_nearest_free_space(map_data, frontier):
    width = map_data.info.width
    height = map_data.info.height

    map_data_array = np.array(map_data.data).reshape((height, width))

    search_radius = 2  # Adjust search radius as needed

    # Convert frontier point to map coordinates
    map_x, map_y = world_to_map_coords(map_data, frontier[0], frontier[1])

    # Generate the sequence of i and j values
    i_values = list(range(0, search_radius + 1)) + list(range(-1, -search_radius - 1, -1))
    j_values = list(range(0, search_radius + 1)) + list(range(-1, -search_radius - 1, -1))

    for i in i_values:
        for j in j_values:
            x = map_x + i
            y = map_y + j

            # Check if the indices are within the bounds of the map
            if 0 <= x < width and 0 <= y < height:
                if map_data_array[y, x] == 0:
                    # Convert back to world coordinates
                    world_x, world_y = map_to_world_coords(map_data, x, y)
                    return [world_x, world_y], True

    # If no free space is found within the radius, return the original frontier
    return frontier, False


def add_free_space_at_robot(map_data, robot_x, robot_y, free_space_radius):
    # Convert radius from world coordinates to map coordinates
    free_space_radius = int(free_space_radius / map_data.info.resolution)
    width = map_data.info.width
    height = map_data.info.height

    # Convert map data to a numpy array
    map_data_array = np.array(map_data.data).reshape((height, width))

    # Convert robot world coordinates to map coordinates
    robot_map_x, robot_map_y = world_to_map_coords(map_data, robot_x, robot_y)

    # Update map data within the radius around the robot
    for i in range(-free_space_radius, free_space_radius + 1):
        for j in range(-free_space_radius, free_space_radius + 1):
            x = robot_map_x + i
            y = robot_map_y + j

            # Check if the indices are within the bounds of the map
            if 0 <= x < width and 0 <= y < height:
                map_data_array[y, x] = 0  # Assuming '0' represents free space

    # Flatten the numpy array and convert it to a list of int8
    flattened_data = map_data_array.flatten().astype(np.int8).tolist()

    # Update the map_data with the modified map
    map_data.data = flattened_data
    return map_data


def is_path_through_walls(map_data, path):
    width = map_data.info.width
    height = map_data.info.height
    reshaped_map = np.array(map_data.data).reshape(height, width)

    for p in path:
        x, y = p
        if 0 <= x < width and 0 <= y < height:
            if reshaped_map[y, x] == 100:
                return True

    return False