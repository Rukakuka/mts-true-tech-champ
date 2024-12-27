from pynput import keyboard

controller = keyboard.Controller()

def button(key, status):
    if status:
        controller.press(key)
    else:
        controller.release(key)


def on_press(key):
    status = True
    try:
        if key == keyboard.Key.up:
            button('w', status)
            button('e', status)
        elif key == keyboard.Key.down:
            button('s', status)
            button('d', status)
        elif key == keyboard.Key.right:
            button('w', status)
            button('d', status)
        elif key == keyboard.Key.left:
            button('e', status)
            button('s', status)
    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.esc:
        # Остановить слушатель
        return False
    
    status = False
    try:
        if key == keyboard.Key.up:
            button('w', status)
            button('e', status)
        elif key == keyboard.Key.down:
            button('s', status)
            button('d', status)
        elif key == keyboard.Key.right:
            button('w', status)
            button('d', status)
        elif key == keyboard.Key.left:
            button('e', status)
            button('s', status)
    except AttributeError:
        pass
    
    

# Начать слушать нажатия клавиш
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()