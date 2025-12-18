import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

# Enable both depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert color from RGB to BGR for OpenCV
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Apply colormap on depth image (to make it visually appealing)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Resize depth_colormap to match color_image for stacking
        depth_colormap_resized = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]))

        # Stack both images horizontally
        images = depth_colormap_resized

        # Display the images
        cv2.imshow('Depth', images)

        # Press ESC to quit
        if cv2.waitKey(1) == 27:
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
