#%%
import cv2
import sys
import numpy as np
from PIL import Image
import os
import cmapy
from argparse import ArgumentParser
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.scene_3d_infer import Scene3DNetworkInfer


def main(): 
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", required=True)
    parser.add_argument("-i", "--video_filepath", dest="video_filepath", required=True)
    parser.add_argument("-o", "--output_file", dest="output_file", required=True)
    parser.add_argument("--skip_frames", type=int, default=0, help="Number of frames to skip between processed frames")
    args = parser.parse_args() 

    # Load model
    model = Scene3DNetworkInfer(checkpoint_path=args.model_checkpoint_path)
    print('Scene3D Model Loaded')
    
    # Open video input
    cap = cv2.VideoCapture(args.video_filepath)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Output video file
    output_filepath_obj = args.output_file + '.avi'
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    skip = args.skip_frames + 1
    fps_output = fps_input / skip  # adjust output FPS for skipped frames

    # Create writer
    writer_obj = cv2.VideoWriter(
        output_filepath_obj,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps_output,
        (1280, 720)
    )

    alpha = 0.97  # transparency factor
    print('Processing started')

    total_frames = 0
    frame_count = 0
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Frame not read - ending processing')
            break

        total_frames += 1
        if total_frames % skip != 1:
            continue  # skip frames

        # Convert frame for model
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb).resize((640, 320))

        # Run inference
        prediction = model.inference(image_pil)
        prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]))

        # Normalize depth map
        prediction_image = 255.0 * ((prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction)))
        prediction_image = prediction_image.astype(np.uint8)

        # Create colored depth map
        vis_obj = cv2.applyColorMap(prediction_image, cmapy.cmap('viridis'))

        # Resize for saving
        frame_hd = cv2.resize(frame, (1280, 720))
        vis_obj_hd = cv2.resize(vis_obj, (1280, 720))

        # Blend visualization
        image_vis_obj = cv2.addWeighted(vis_obj_hd, alpha, frame_hd, 1 - alpha, 0)

        # Save frame
        writer_obj.write(image_vis_obj)

        # === SHOW LIVE PREVIEW AT 640x480 ===
        preview = cv2.resize(image_vis_obj, (640, 480))
        cv2.imshow("Scene3D Depth Visualization (Live)", preview)

        # FPS monitor
        frame_count += 1
        t = time.time()
        if t - prev_time >= 1.0:
            print(f"FPS (preview): {frame_count}")
            frame_count = 0
            prev_time = t

        # Quit preview with Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer_obj.release()
    cv2.destroyAllWindows()
    print('Completed')


if __name__ == '__main__':
    main()
# %%

