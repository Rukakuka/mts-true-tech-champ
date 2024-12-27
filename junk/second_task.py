import numpy as np
import requests
import time
from collections import deque

# Constants
API_BASE_URL = "http://127.0.0.1:8801/api/v1"
TOKEN = "44f20028-0138-410c-a1cd-c660dc8d15c466cb1331-67e8-4897-a192-c5ef202068e2"
MAZE_SIZE = 16
WALL_THRESHOLD = 65
CENTER_COORDINATES = ([7, 7], [7, 8], [8, 7], [8, 8])
WALLS_DICT = {
        0: {
            'W': False,
            'N': False,
            'E': False,
            'S': False 
        },
        1: {
            'W': True,
            'N': False,
            'E': False,
            'S': False 
        },
        2: {
            'W': False,
            'N': True,
            'E': False,
            'S': False 
        },
        3: {
            'W': False,
            'N': False,
            'E': True,
            'S': False 
        },
        4: {
            'W': False,
            'N': False,
            'E': False,
            'S': True 
        },
        5: {
            'W': True,
            'N': False,
            'E': False,
            'S': True 
        },
        6: {
            'W': False,
            'N': False,
            'E': True,
            'S': True 
        },
        7: {
            'W': False,
            'N': True,
            'E': True,
            'S': False 
        },
        8: {
            'W': True,
            'N': True,
            'E': False,
            'S': False 
        },
        9: {
            'W': True,
            'N': False,
            'E': True,
            'S': False 
        },
        10: {
            'W': False,
            'N': True,
            'E': False,
            'S': True 
        },
        11: {
            'W': False,
            'N': True,
            'E': True,
            'S': True 
        },
        12: {
            'W': True,
            'N': True,
            'E': True,
            'S': False 
        },
        13: {
            'W': True,
            'N': True,
            'E': False,
            'S': True 
        },
        14: {
            'W': True,
            'N': False,
            'E': True,
            'S': True 
        },
        15: {
            'W': True,
            'N': True,
            'E': True,
            'S': True 
        }
    }

# Current robot position (starts at bottom-left corner)
current_position = [MAZE_SIZE - 1, 0]
current_orientation = 'N'

# API functions
def get_sensor_data():
    response = requests.get(f"{API_BASE_URL}/robot-cells/sensor-data?token={TOKEN}")
    return response.json()

def move_forward():
    requests.post(f"{API_BASE_URL}/robot-cells/forward?token={TOKEN}")

def move_backward():
    requests.post(f"{API_BASE_URL}/robot-cells/backward?token={TOKEN}")

def turn_right():
    requests.post(f"{API_BASE_URL}/robot-cells/right?token={TOKEN}")

def turn_left():
    requests.post(f"{API_BASE_URL}/robot-cells/left?token={TOKEN}")

def send_answer(data: list):
    response = requests.post(
        url=f"{API_BASE_URL}/matrix/send?token={TOKEN}",
        json=data
    )
    print(response)

def restart_maze():
    response = requests.post(f"{API_BASE_URL}/maze/restart?token={TOKEN}")
    print(response)

# Helper functions (mostly unchanged)
def normalize_angle(angle):
    return (angle + 360) % 360

def get_orientation(yaw):
    normalized_yaw = normalize_angle(yaw)
    if 315 <= normalized_yaw < 360 or 0 <= normalized_yaw < 45:
        return 'N'
    elif 45 <= normalized_yaw < 135:
        return 'E'
    elif 135 <= normalized_yaw < 225:
        return 'S'
    elif 225 <= normalized_yaw < 315:
        return 'W'

def update_position(move):
    global current_position, current_orientation
    new_position = current_position.copy()
    
    if current_orientation == 'N' and move == 'right':
        new_position[1] += 1
    elif current_orientation == 'N' and move == 'forward':
        new_position[0] -= 1      

    elif current_orientation == 'E' and move == 'right':
        new_position[0] += 1
    elif current_orientation == 'E' and move == 'forward':
        new_position[1] += 1

    elif current_orientation == 'S' and move == 'right':
        new_position[1] -= 1
    elif current_orientation == 'S' and move == 'forward':
        new_position[0] += 1

    elif current_orientation == 'W' and move == 'right':
        new_position[0] -= 1
    elif current_orientation == 'W' and move == 'forward':
        new_position[1] -= 1
    
    if 0 <= new_position[0] < MAZE_SIZE and 0 <= new_position[1] < MAZE_SIZE:
        current_position = new_position
    else:
        print(f"Warning: Attempted to move out of maze bounds. Current position: {current_position}")

def detect_walls(sensor_data):
    front = sensor_data['front_distance'] < WALL_THRESHOLD
    right = sensor_data['right_side_distance'] < WALL_THRESHOLD
    left = sensor_data['left_side_distance'] < WALL_THRESHOLD
    back = sensor_data['back_distance'] < WALL_THRESHOLD
    
    walls = {
        'N': 0,
        'E': 0,
        'S': 0,
        'W': 0
    }
    
    if current_orientation == 'N':
        walls['N'], walls['E'], walls['S'], walls['W'] = front, right, back, left
    elif current_orientation == 'E':
        walls['N'], walls['E'], walls['S'], walls['W'] = left, front, right, back
    elif current_orientation == 'S':
        walls['N'], walls['E'], walls['S'], walls['W'] = back, left, front, right
    elif current_orientation == 'W':
        walls['N'], walls['E'], walls['S'], walls['W'] = right, back, left, front
    
    return walls

def calculate_cell_value(walls):
    if not walls['W'] and not walls['N'] and not walls['E'] and not walls['S']:
        return 0
    elif walls['W'] and not walls['N'] and not walls['E'] and not walls['S']:
        return 1
    elif not walls['W'] and walls['N'] and not walls['E'] and not walls['S']:
        return 2
    elif not walls['W'] and not walls['N'] and walls['E'] and not walls['S']:
        return 3
    elif not walls['W'] and not walls['N'] and not walls['E'] and walls['S']:
        return 4
    elif walls['W'] and not walls['N'] and not walls['E'] and walls['S']:
        return 5
    elif not walls['W'] and not walls['N'] and walls['E'] and walls['S']:
        return 6
    elif not walls['W'] and walls['N'] and walls['E'] and not walls['S']:
        return 7
    elif walls['W'] and walls['N'] and not walls['E'] and not walls['S']:
        return 8
    elif walls['W'] and not walls['N'] and walls['E'] and not walls['S']:
        return 9
    elif not walls['W'] and walls['N'] and not walls['E'] and walls['S']:
        return 10
    elif not walls['W'] and walls['N'] and walls['E'] and walls['S']:
        return 11
    elif walls['W'] and walls['N'] and walls['E'] and not walls['S']:
        return 12
    elif walls['W'] and walls['N'] and not walls['E'] and walls['S']:
        return 13
    elif walls['W'] and not walls['N'] and walls['E'] and walls['S']:
        return 14
    elif walls['W'] and walls['N'] and walls['E'] and walls['S']:
        return 15

def update_maze(walls, maze):
    global current_position
    cell_value = calculate_cell_value(walls)
    if 0 <= current_position[0] < MAZE_SIZE and 0 <= current_position[1] < MAZE_SIZE:
        maze[current_position[0], current_position[1]] = cell_value
    else:
        print(f"Error: Attempted to update cell outside maze bounds. Position: {current_position}")

def move_robot(sensor_data):
    # Right-hand rule logic
    if sensor_data['right_side_distance'] > WALL_THRESHOLD:
        turn_right()
        move_forward()
        return "right"
    elif sensor_data['front_distance'] > WALL_THRESHOLD:
        move_forward()
        return "forward"
    else:
        turn_left()
        return "left"

# Pathfinding functions
def find_exit(last_visited_node: tuple):
    if last_visited_node == (7, 6) or last_visited_node == (6, 7):
        return (7, 7)
    elif last_visited_node == (6, 8) or last_visited_node == (7, 9):
        return (7, 8)
    elif last_visited_node == (8, 9) or last_visited_node == (10, 8):
        return (8, 8)
    elif last_visited_node == (9, 8) or last_visited_node == (8, 6):
        return (8, 7)
    # exit_ids = {1, 2, 3, 4}
    # for exit_tile_coordinate in CENTER_COORDINATES:
    #     if maze[exit_tile_coordinate[0], exit_tile_coordinate[1]] in exit_ids:
    #         return exit_tile_coordinate

def get_neighbors(cell, maze):
    x, y = cell
    neighbors = []
    directions = [(0, 1, 'E'), (1, 0, 'S'), (0, -1, 'W'), (-1, 0, 'N')]
    cell_value = maze[x][y]
    
    for dx, dy, direction in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < MAZE_SIZE and 0 <= ny < MAZE_SIZE:
            if cell_value != -1 and not has_wall(cell_value, direction):
                neighbors.append((nx, ny))
    
    return neighbors

def has_wall(cell_value, direction):
    return WALLS_DICT[cell_value][direction]

def bfs_shortest_path(start, goal, maze):
    queue = deque([[start]])
    visited = set([start])
    
    while queue:
        path = queue.popleft()
        cell = path[-1]
        
        if cell == tuple(goal):
            return path
        
        for neighbor in get_neighbors(cell, maze):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    
    return None

def calculate_path_cost(path):
    cost = 0
    current_direction = 'N'  # Assuming the robot starts facing South
    
    for i in range(1, len(path)):
        prev_cell, curr_cell = path[i-1], path[i]
        dx, dy = curr_cell[0] - prev_cell[0], curr_cell[1] - prev_cell[1]
        
        if dx == 1:
            new_direction = 'S'
        elif dx == -1:
            new_direction = 'N'
        elif dy == 1:
            new_direction = 'E'
        else:
            new_direction = 'W'
        
        if new_direction != current_direction:
            cost += 1.5  # Cost of turning
        
        cost += 1  # Cost of moving forward
        current_direction = new_direction
    
    return cost

def navigate_to_center(path):
    global current_position, current_orientation
    
    orientations = ['N', 'E', 'S', 'W']
    
    for i in range(1, len(path)):
        prev_cell, curr_cell = path[i-1], path[i]
        dx, dy = curr_cell[0] - prev_cell[0], curr_cell[1] - prev_cell[1]
        
        if dx == 1:
            target_orientation = 'S'
        elif dx == -1:
            target_orientation = 'N'
        elif dy == 1:
            target_orientation = 'E'
        else:
            target_orientation = 'W'
        
        # Determine the most efficient way to turn
        current_index = orientations.index(current_orientation)
        target_index = orientations.index(target_orientation)
        
        turn_difference = (target_index - current_index + 4) % 4
        
        if turn_difference == 0:
            # Already facing the right direction, no need to turn
            pass
        elif turn_difference == 1:
            turn_right()
        elif turn_difference == 2:
            # It's faster to move backward
            pass
        else:  # turn_difference == 3
            turn_left()
        
        current_orientation = target_orientation
        
        # Move in the correct direction
        if turn_difference == 2:
            move_backward()
        else:
            move_forward()
        
        current_position = list(curr_cell)
        time.sleep(0.1)
    
    print(f"Reached the center at {current_position}")

# Make sure to add these functions if they're not already present:

def turn_left():
    global current_orientation
    requests.post(f"{API_BASE_URL}/robot-cells/left?token={TOKEN}")
    current_orientation = {'N': 'W', 'E': 'N', 'S': 'E', 'W': 'S'}[current_orientation]
    time.sleep(0.1)

def move_backward():
    requests.post(f"{API_BASE_URL}/robot-cells/backward?token={TOKEN}")
    time.sleep(0.1)

def fullfil_maze(maze, last_visited_node):
    maze[7, 7] = 8
    maze[7, 8] = 7
    maze[8, 8] = 6
    maze[8, 7] = 5
    if last_visited_node == (7, 6):
        maze[7, 7] = 2
    elif last_visited_node == (6, 7):
        maze[7, 7] = 1
    elif last_visited_node == (6, 8):
        maze[7, 8] = 3
    elif last_visited_node == (7, 9):
        maze[7, 8] = 2
    elif last_visited_node == (8, 9):
        maze[8, 8] = 4
    elif last_visited_node == (10, 8):
        maze[8, 8] = 3
    elif last_visited_node == (9, 8):
        maze[8, 7] = 1
    elif last_visited_node == (8, 6):
        maze[8, 7] = 4
    return maze


def main():
    global current_orientation, current_position
    steps = 0
    max_steps = MAZE_SIZE * MAZE_SIZE * 4
    # Initialize maze array
    maze = np.full((MAZE_SIZE, MAZE_SIZE), -1, dtype=int)

    # Phase 1: Research the labyrinth
    if True:
        while -1 in maze and steps < max_steps:
            if current_position in CENTER_COORDINATES:
                break
            sensor_data = get_sensor_data()
            # print('Sensors:', sensor_data)
            current_orientation = get_orientation(sensor_data['rotation_yaw'])
            # print('Orientation:', current_orientation)
            walls = detect_walls(sensor_data)
            # print('Walls:', walls)
            update_maze(walls, maze)
            # print('Maze:', maze)
            last_visited_node = current_position
            move = move_robot(sensor_data)
            # print('Move:', (move))
            update_position(move)
            # print('Orientation:', current_orientation, 'Position:', current_position)
            steps += 1

        if steps >= max_steps:
            print("Maximum number of steps reached. Robot might be stuck in a loop.")
        else:
            print("Exit is found")

        print("Maze configuration:")
        print(maze)

    # maze = np.array(
    #         [[12,  8,  2, 11,  8,  7,  8, 10, 10, 10,  7,  8, 10, 10,  2,  7],
    #         [ 5,  6,  1,  7,  9,  5,  6,  8, 10, 10,  3,  9, 13,  7,  9, 14],
    #         [ 8,  7,  9,  9,  5,  7,  8,  6,  8,  7,  9,  5,  7,  9,  5,  7],
    #         [ 9,  9, 14,  5, 10,  6,  9,  8,  6,  9, 14,  8,  6,  1,  7,  9],
    #         [ 9,  5,  7,  8, 10,  2,  6,  9, 12,  5, 10,  6,  8,  6,  9,  9],
    #         [ 1,  7,  9,  9,  8,  6,  8,  6,  1, 10,  2,  7,  1,  7,  9,  9],
    #         [ 9,  9,  5,  6,  5,  7,  9, 13,  4, 11,  9,  9, 14,  9, 14,  9],
    #         [ 9,  5,  7,  8,  7,  9,  1, -1, -1,  8,  6,  9,  8,  3,  8,  3],
    #         [ 5,  7,  9, 14,  5,  6,  9, -1, -1,  9, 12,  5,  6,  9,  9, 14],
    #         [12,  9,  9,  8, 10, 10,  4,  7,  8,  6,  5,  2, 10,  6,  5,  7],
    #         [ 9,  9,  9,  9,  8, 10,  7,  9,  5, 10,  7,  9,  8,  7,  8,  6],
    #         [ 9,  9,  9,  9,  9, 12,  9,  5, 10, 10,  6,  9,  9,  9,  5,  7],
    #         [ 5,  6,  5,  6,  9,  9,  9,  8, 10,  7, 12,  1,  6,  5, 11,  9],
    #         [ 8, 10, 10, 10,  6,  1,  6,  9, 12,  9,  5,  6,  8, 10,  7,  9],
    #         [ 9,  8, 10, 10, 10,  6, -1,  3,  9,  5,  7,  8,  6, 12,  5,  3],
    #         [14,  5, 10, 10, 10, 10, 10,  6,  5, 10,  4,  6, 13,  4, 10,  6]]
    #     )
    # print("Maze configuration:")
    # print(maze)

    # last_visited_node = (7, 6)
    maze = fullfil_maze(maze, last_visited_node)
    print(maze)
    print("last_visited_node: ", last_visited_node)
    
    # Phase 2: Restart the maze
    print("Restarting the maze...")
    restart_maze()
    time.sleep(0.5)  # Give some time for the restart to take effect

    # Reset the robot's position and orientation
    current_position = [MAZE_SIZE - 1, 0]
    current_orientation = 'N'

    # Phase 3: Find the fastest way to the center and navigate to the center
    exit = find_exit(tuple(last_visited_node))
    print(exit)

    path = bfs_shortest_path((15, 0), exit, maze)
    if path is None:
        print("No path to center found!")
        return

    # cost = calculate_path_cost(path) # Скорее всего он неправильно кост считает. Но вроде пока и не нужно.
    print(f"Shortest path to center: {path}")
    # print(f"Path cost: {cost}")

    print("Navigating to the center...")
    navigate_to_center(path)

if __name__ == "__main__":
    main()