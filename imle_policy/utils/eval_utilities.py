import cv2
import os
from PIL import Image
import numpy as np
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import time

import pdb

import argparse






def save_start_states():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_device('035122250388')  # choose specific camera

    # Start streaming
    profile = pipeline.start(config)

    # Get the color sensor
    color_sensor = profile.get_device().query_sensors()[1]  # Typically, index 1 is for the color sensor
    color_sensor.set_option(rs.option.exposure, 70)  # Set exposure to 70
    color_sensor.set_option(rs.option.gain, 70)      # Set gain to 70

    time.sleep(2)

    # flusha few frames to allow the camera to adjust to the new settings
    for i in range(10):
        pipeline.wait_for_frames()

    try:
        for i in range(20):
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Display the image
            plt.imshow(color_image)
            plt.show()
            cv2.imwrite(f"start_states/{i}.png", color_image)
    finally:
        pipeline.stop()



def overlay_image_on_realsense_stream(image_path, alpha=0.5, device_id='035122250388'):
    """
    Overlays a single image onto a RealSense camera stream in real-time.

    Parameters:
        image_path (str): Path to the image file to be overlayed.
        alpha (float): Transparency level of the overlay image (0.0 to 1.0).
        device_id (str): Device ID of the RealSense camera.
    """
    # Load the overlay image with potential alpha channel
    overlay_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Check if the image has an alpha channel
    if overlay_image.shape[2] == 4:
        # Split overlay image into color and alpha channels
        overlay_bgr = overlay_image[..., :3]  # Color channels
        overlay_alpha = overlay_image[..., 3] / 255.0  # Normalize alpha channel to 0-1 range
    else:
        # If no alpha channel, use a fully opaque mask
        overlay_bgr = overlay_image
        overlay_alpha = np.ones(overlay_bgr.shape[:2], dtype=np.float32)

    # Set up RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_device(device_id)

    # Start streaming
    profile = pipeline.start(config)

    # Set camera options
    color_sensor = profile.get_device().query_sensors()[1]
    color_sensor.set_option(rs.option.exposure, 70)
    color_sensor.set_option(rs.option.gain, 70)

    # Allow camera to adjust to new settings
    time.sleep(2)
    for _ in range(10):
        pipeline.wait_for_frames()

    try:
        while True:
            # Get the latest frames from the camera
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert the frame to a numpy array
            frame = np.asanyarray(color_frame.get_data())

            # Resize overlay image and alpha channel to match the frame size
            overlay_resized = cv2.resize(overlay_bgr, (frame.shape[1], frame.shape[0]))
            overlay_alpha_resized = cv2.resize(overlay_alpha, (frame.shape[1], frame.shape[0]))

            # Expand overlay_alpha_resized to match the 3-channel shape of frame
            overlay_alpha_resized = cv2.merge([overlay_alpha_resized] * 3)

            # Blend overlay with the frame using alpha blending
            blended_frame = (frame * (1 - overlay_alpha_resized * alpha) +
                             overlay_resized * (overlay_alpha_resized * alpha)).astype(np.uint8)

            # Display the blended frame
            cv2.imshow("Overlayed RealSense Stream", blended_frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()



if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser(description='Overlay an image on a RealSense camera stream.')
    parser.add_argument('--idx', type=int, default=0, help='Index of the image to overlay.')

    idx = parser.parse_args().idx

    overlay_image_on_realsense_stream(f"start_states/{parser.parse_args().idx}.png", alpha=0.5, device_id='035122250388')

    # save_start_states()