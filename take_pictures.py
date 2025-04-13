import cv2
from picamera2 import Picamera2
import time
import os
from datetime import datetime

# Initialize Picamera2
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (512, 512)})
picam2.configure(camera_config)
picam2.start()

title = "Live image"
save_path = "/home/rspi4/rspi-scripts/images"

if not os.path.exists(save_path):
    os.makedirs(save_path)

cv2.namedWindow(title, cv2.WINDOW_NORMAL)

update_image = False

def on_mouse_click(event, x, y, flags, param):
    global update_image
    if event == cv2.EVENT_LBUTTONDOWN:
        update_image = True
cv2.setMouseCallback(title, on_mouse_click)

time.sleep(1)  # Allow camera to warm up

while True:
    try:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(cv2.flip(frame, -1), cv2.COLOR_RGB2BGR)

        if update_image:
            fn = f"{save_path}/img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            print(f"Saving {fn}")
            cv2.imwrite(fn, frame)
            update_image = False  # Wait for the next click to update

        cv2.imshow(title, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break


    except KeyboardInterrupt:
        print("Interrupted by user.")
        break

    except Exception as e:
        print(f"An error occurred: {e}")
        break

cv2.destroyAllWindows()
picam2.stop()
