from PIL import Image
import numpy as np
import io

# Reload the uploaded image directly from the conversation
input_path = "F:\lidar_ext\img\IMG-20250727-WA0001.jpg"
output_path = "vr_side_by_side.jpg"

# Open the image
img = Image.open(input_path).convert("RGB")
w, h = img.size

# Stereo shift in pixels
shift_px = 40

# Convert image to arrays
left_array = np.array(img)
right_array = np.array(img)

# Apply shift for left/right images
left_shifted = np.zeros_like(left_array)
right_shifted = np.zeros_like(right_array)

left_shifted[:, :-shift_px] = left_array[:, shift_px:]
right_shifted[:, shift_px:] = right_array[:, :-shift_px]

# Convert arrays back to images
left_img = Image.fromarray(left_shifted)
right_img = Image.fromarray(right_shifted)

# Merge side by side
sbs = Image.new("RGB", (w * 2, h))
sbs.paste(left_img, (0, 0))
sbs.paste(right_img, (w, 0))

# Save the final JPEG
sbs.save(output_path, "JPEG")
output_path
