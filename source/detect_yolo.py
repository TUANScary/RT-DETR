import cv2
import time
import os
from ultralytics import YOLO

# Load trained YOLOv8 model
model_path = r"model\yolov8\best.pt"  
model = YOLO(model_path)

# Ask for input path
input_path = input("Enter the path of the image or video: ")

# Check if input is a video
if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Integer FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    prev_time = 0
    
    # Ask user if they want to save the output
    save_output = input("Do you want to save the processed video? (y/n): ").strip().lower()
    if save_output == 'y':
        output_filename = input("Enter the output video filename (with extension, e.g., output.mp4): ")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Start timer to calculate FPS
        start_time = time.time()

        # Run YOLO detection
        results = model(frame)
        
        # Draw detection results
        for result in results:
            frame = result.plot()

        # Calculate FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        # Display FPS on video
        cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save processed video if enabled
        if save_output == 'y':
            out.write(frame)
        
        # Show output
        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    if save_output == 'y':
        out.release()
    cv2.destroyAllWindows()
    
else:
    # If it's an image
    image = cv2.imread(input_path)
    if image is None:
        print("Error: Could not open image.")
        exit()

    # Run YOLO detection
    results = model(image)

    # Draw detection results
    for result in results:
        image = result.plot()
    
    # Show image
    cv2.imshow("YOLOv8 Detection", image)
    
    # Ask user if they want to save the output image
    save_output = input("Do you want to save the processed image? (y/n): ").strip().lower()
    if save_output == 'y':
        output_filename = input("Enter the output image filename (with extension, e.g., output.jpg): ")
        cv2.imwrite(output_filename, image)
        print(f"Image saved as {output_filename}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

