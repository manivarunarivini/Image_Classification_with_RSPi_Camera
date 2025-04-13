import numpy as np
import cv2
import onnxruntime as ort
from picamera2 import Picamera2
import time
from custom_fun import draw_text
import sys

stream = "--stream" in sys.argv

# Initialize Picamera2
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (448, 448)})
picam2.configure(camera_config)
picam2.start()

# Load the ONNX model
onnx_model_path = "/home/rspi4/rspi-scripts/models/2024-10-23_Image_Classification.onnx"
session = ort.InferenceSession(onnx_model_path)

# Preprocessing function for AlexNet
def preprocess_image(image):
    # Resize to 224x224
    image = cv2.resize(image, (224, 224))
    # Convert image to float32 and normalize
    image = image.astype(np.float32) / 255.0  # Ensure image is float32
    # Transpose from (H, W, C) to (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    # Normalize the image using the same mean and std as during training
    image = (image - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(3, 1,
                                                                                                                 1)
    # Add batch dimension
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Ensure image is float32
    return image


# Class labels corresponding to your leaves dataset
class_labels = [
    "Fern",  # Replace with your actual class names
    "Ginkgo",
    "Ivy",
    "Kummerowia striata",
    "Laciniata",
    "Macrolobium acaciifolium",
    "Micranthes odontoloma",
    "Murraya",
    "No Leaf",
    "Robinia pseudoacacia",
    "Selaginella davidi franch",
]

update_image = True
title_suffix = "stream" if stream else "click"
title = f"Leafs AlexNet ({title_suffix})"

cv2.namedWindow(title)

if not stream:
    def on_mouse_click(event, x, y, flags, param):
        global update_image
        if event == cv2.EVENT_LBUTTONDOWN:
            update_image = True
    cv2.setMouseCallback(title, on_mouse_click)

time.sleep(1)  # Allow camera to warm up
while True:
    try:
        if stream or update_image:
            # Capture an image using Picamera2
            frame = picam2.capture_array()
            frame = cv2.flip(frame, -1)

            # Apply transformations
            inp = preprocess_image(np.array(frame))

            # Perform inference with ONNX Runtime
            ort_inputs = {session.get_inputs()[0].name: inp}
            ort_outs = session.run(None, ort_inputs)

            # Process the output
            out = ort_outs[0]  # Get the output from the model (logits or probabilities)
            probabilities = np.exp(out) / np.sum(np.exp(out))  # Apply softmax to get probabilities

            predicted_class_index = np.argmax(probabilities, axis=1)[0]  # Get the index of the highest score
            predicted_probability = probabilities[0][predicted_class_index]  # Get the highest probability
            predicted_class = class_labels[predicted_class_index]

            # Only print if the prediction probability is greater than 50%
            if predicted_probability < 0.5 or predicted_class == "No Leaf":
                col = (0, 0, 255)
                text = "No Leafs Detected"
                print(text)
            else:
                col = (0, 255, 0)
                text = f"{predicted_class}: {predicted_probability:.2f}"
                print(text)

            # Optionally visualize the prediction result
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = draw_text(img, text, text_color=col)
            cv2.imshow(title, img)
            update_image = False  # Wait for the next click to update

        time.sleep(0.5)

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
