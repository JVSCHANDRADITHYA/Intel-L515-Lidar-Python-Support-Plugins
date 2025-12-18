from ultralytics import YOLO
import cv2

# Load a YOLOv8 model (use 'yolov8n.pt' for a lightweight version)
model = YOLO("pcd_model_pillar.pt") # or 'yolov8s.pt', 'yolov8m.pt', etc.

# Open webcam (0 = default webcam)
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and cleanup
cap.release()
cv2.destroyAllWindows()
