import numpy as np
import requests
import time

# Константы
API_BASE_URL = "http://127.0.0.1:8801/api/v1/robot-cells"
TOKEN = "56187060-fd11-4b07-8efa-e662d71dc7f8f6dd1a94-7c0d-497a-a83b-d7f91be8ca60"
MAZE_SIZE = 16
WALL_THRESHOLD = 65

# Инициализация массива лабиринта
maze = np.full((MAZE_SIZE, MAZE_SIZE), -1, dtype=int)

# Текущая позиция робота (начинает с нижнего левого угла)
current_position = [MAZE_SIZE - 1, 0]

def get_sensor_data():
    response = requests.get(f"{API_BASE_URL}/sensor-data?token={TOKEN}")
    return response.json()

def move_forward():
    requests.post(f"{API_BASE_URL}/forward?token={TOKEN}")

def move_backward():
    requests.post(f"{API_BASE_URL}/backward?token={TOKEN}")

def turn_right():
    requests.post(f"{API_BASE_URL}/right?token={TOKEN}")

def turn_left():
    requests.post(f"{API_BASE_URL}/left?token={TOKEN}")

def normalize_angle(angle):
    return (angle + 360) % 360

def get_orientation(yaw):
    normalized_yaw = normalize_angle(yaw)
    # print(normalized_yaw)
    if 315 <= normalized_yaw < 360 or 0 <= normalized_yaw < 45:
        return 'N'
    elif 45 <= normalized_yaw < 135:
        return 'E'
    elif 135 <= normalized_yaw < 225:
        return 'S'
    elif 225 <= normalized_yaw < 315:
        return 'W'

def update_position(move, orientation):
    global current_position
    new_position = current_position.copy()
    if orientation == 'N' and move == 'right':
        new_position[1] += 1
    elif orientation == 'N' and move == 'forward':
        new_position[0] -= 1      

    elif orientation == 'E' and move == 'right':
        new_position[0] += 1
    elif orientation == 'E' and move == 'forward':
        new_position[1] += 1

    elif orientation == 'S' and move == 'right':
        new_position[1] -= 1
    elif orientation == 'S' and move == 'forward':
        new_position[0] += 1

    elif orientation == 'W' and move == 'right':
        new_position[0] -= 1
    elif orientation == 'W' and move == 'forward':
        new_position[1] -= 1
    
    # Проверка границ
    if 0 <= new_position[0] < MAZE_SIZE and 0 <= new_position[1] < MAZE_SIZE:
        current_position = new_position
    else:
        print(f"Предупреждение: попытка выйти за границы лабиринта. Текущая позиция: {current_position}")

def detect_walls(sensor_data, orientation):
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
    
    if orientation == 'N':
        walls['N'], walls['E'], walls['S'], walls['W'] = front, right, back, left
    elif orientation == 'E':
        walls['N'], walls['E'], walls['S'], walls['W'] = left, front, right, back
    elif orientation == 'S':
        walls['N'], walls['E'], walls['S'], walls['W'] = back, left, front, right
    elif orientation == 'W':
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

def update_maze(walls):
    global maze, current_position
    cell_value = calculate_cell_value(walls)
    if 0 <= current_position[0] < MAZE_SIZE and 0 <= current_position[1] < MAZE_SIZE:
        maze[current_position[0], current_position[1]] = cell_value
    else:
        print(f"Ошибка: попытка обновить ячейку за пределами лабиринта. Позиция: {current_position}")

def move_robot(sensor_data):
    # Логика обхода по правилу правой руки
    if sensor_data['right_side_distance'] > 65:  # Если справа свободно
        turn_right()
        move_forward()
        return "right"
    elif sensor_data['front_distance'] > 65:  # Если впереди свободно
        move_forward()
        return "forward"
    else:
        turn_left()  # Поворачиваем налево если впереди и справа есть стены
        return "left"

def main():
    steps = 0
    max_steps = MAZE_SIZE * MAZE_SIZE * 4  # Предельное количество шагов

    while -1 in maze and steps < max_steps:
        # print(maze)
        sensor_data = get_sensor_data()
        # print(sensor_data)
        orientation = get_orientation(sensor_data['rotation_yaw'])
        # print(orientation)
        walls = detect_walls(sensor_data, orientation)
        print(walls)

        update_maze(walls)
        print(maze)
        move = move_robot(sensor_data)
        update_position(move, orientation)
        
        steps += 1
        time.sleep(0.5)  # Небольшая задержка для стабильности
    
    if steps >= max_steps:
        print("Достигнуто максимальное количество шагов. Возможно, робот зациклился.")
    else:
        print("Лабиринт полностью исследован!")
    
    print(maze)

if __name__ == "__main__":
    main()