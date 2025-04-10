import cv2
import time
import torch
import os
import numpy as np
from PIL import Image
from transformers import AutoModelForObjectDetection, AutoImageProcessor

# Load RT-DETR model
CHECKPOINT = r"model\rt-detr\checkpoint-3116"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForObjectDetection.from_pretrained(CHECKPOINT).to(DEVICE)
processor = AutoImageProcessor.from_pretrained(CHECKPOINT)

# Get class labels from model
CLASS_LABELS = model.config.id2label

BBOX_COLOR = (0, 255, 0)  

def detect_objects(image):
    """Runs RT-DETR on an image and returns an annotated image."""
    inputs = processor(image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    w, h = image.size

    results = processor.post_process_object_detection(
        outputs, target_sizes=[(h, w)], threshold=0.3
    )[0]

    # Convert image to OpenCV format
    draw = np.array(image)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box]
        class_name = CLASS_LABELS.get(label.item(), f"Unknown({label.item()})")

        # Draw bounding box
        cv2.rectangle(draw, (box[0], box[1]), (box[2], box[3]), BBOX_COLOR, 2)
        cv2.putText(draw, f"{class_name}: {score:.2f}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_COLOR, 2)

    return draw


input_path = input("Enter the path of the image or video: ")

# Check if input is a video
if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    prev_time = 0

    save_output = input("Do you want to save the processed video? (y/n): ").strip().lower()
    if save_output == 'y':
        output_filename = input("Enter the output video filename (e.g., output.mp4): ")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Convert OpenCV frame to PIL format
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run object detection
        result_frame = detect_objects(image)

        # Calculate FPS
        curr_time = time.time()
        fps_display = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        # Display FPS
        cv2.putText(result_frame, f"FPS: {fps_display}", (width - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if save_output == 'y':
            out.write(result_frame)

        cv2.imshow("RT-DETR Detection", result_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if save_output == 'y':
        out.release()
    cv2.destroyAllWindows()

else:
    image = cv2.imread(input_path)
    if image is None:
        print("Error: Could not open image.")
        exit()

    # Convert OpenCV image to PIL format
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Run object detection
    result_image = detect_objects(pil_image)

    # Show image
    cv2.imshow("RT-DETR Detection", result_image)

    save_output = input("Do you want to save the processed image? (y/n): ").strip().lower()
    if save_output == 'y':
        output_filename = input("Enter the output image filename (e.g., output.jpg): ")
        cv2.imwrite(output_filename, result_image)
        print(f"Image saved as {output_filename}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
