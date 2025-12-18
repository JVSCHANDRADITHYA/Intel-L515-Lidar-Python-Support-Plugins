import pyrealsense2 as rs
import numpy as np

# Create a pipeline
pipeline = rs.pipeline()

# Configure the pipeline for depth and color streams
config = rs.config()
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

try:
    while True:
        # Wait for a coherent set of frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Process the depth and color images here
        # For example, print the depth value at the center of the image
        # center_x, center_y = depth_image.shape[1] // 2, depth_image.shape[0] // 2
        # distance = depth_frame.get_distance(center_x, center_y)
        # print(f"Distance at center: {distance:.2f} meters")

finally:
    # Stop the pipeline
    pipeline.stop()