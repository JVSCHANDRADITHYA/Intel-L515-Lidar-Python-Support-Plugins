import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLOv8 model (you can use your trained model here)
model = YOLO("pcd_model_pillar.pt")

# RealSense setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
pipeline.start(config)


cv2.namedWindow("YOLO + Depth", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("YOLO + Depth", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# Align depth to color frame
align = rs.align(rs.stream.color)

try:
    while True:
        # Wait for aligned frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Run YOLO on color frame
        results = model(color_image_bgr)

        # Prepare depth image visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # Draw YOLO boxes on depth_colormap
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = box.conf[0]

                color = np.random.randint(0, 255, size=3).tolist()
                
                # Draw box and label
                cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), color, 2)
                cv2.putText(depth_colormap, f"{label} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

        # Show only the annotated depth image
        cv2.imshow("YOLO + Depth", depth_colormap)


        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
