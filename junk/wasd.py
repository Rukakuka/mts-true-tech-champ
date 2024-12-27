import requests
from pynput import keyboard

# Define the base URL and token
BASE_URL = "http://127.0.0.1:8801/api/v1/robot-cells"
TOKEN = "65bcecaf-f60c-4c20-9d32-732576a5b0b39217b0bb-c5ed-4201-8131-65bdbaafa193"

# Function to send POST request
def send_request(direction):
    url = f"{BASE_URL}/{direction}?token={TOKEN}"
    try:
        response = requests.post(url)
        if response.status_code == 200:
            print(f"Successfully sent {direction} command.")
        else:
            print(f"Failed to send {direction} command: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error when sending {direction} command: {e}")

# Define the key press event handler
def on_press(key):
    try:
        if key.char.lower() == 'w':
            send_request('forward')
        elif key.char.lower() == 'a':
            send_request('left')
        elif key.char.lower() == 's':
            send_request('backward')
        elif key.char.lower() == 'd':
            send_request('right')
    except AttributeError:
        pass  # Non-character key pressed

# Define the key release event handler
def on_release(key):
    if key == keyboard.Key.esc:
        return False  # Stop listener

# Start the listener
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()