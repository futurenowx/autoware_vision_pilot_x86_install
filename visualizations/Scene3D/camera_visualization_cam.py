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

# Constants
OUTPUT_SIZE = (1280, 720)
PREVIEW_SIZE = (640, 320)
INF_SIZE = (640, 320)
ALPHA_BLEND = 0.97  # transparency factor

def main(): 
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", required=True)
    parser.add_argument("--camera_id", type=int, default=6)
    parser.add_argument("-o", "--output_file", default=None)
    parser.add_argument("--skip_frames", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args() 

    # Load model
    model = Scene3DNetworkInfer(checkpoint_path=args.model_checkpoint_path)
    print("Scene3D Model Loaded")

    # Open camera
    print(f"Opening camera ID {args.camera_id}...")
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera_id}")
        return

    # Warm up camera
    print("Warming up camera...")
    for _ in range(20):
        cap.read()

    fps_input = cap.get(cv2.CAP_PROP_FPS)
    if fps_input is None or fps_input <= 0:
        fps_input = 30.0
    skip = args.skip_frames + 1
    fps_output = fps_input / skip

    # Optional video writer
    writer_obj = None
    if args.output_file:
        writer_obj = cv2.VideoWriter(
            args.output_file + ".avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps_output,
            OUTPUT_SIZE
        )
        print(f"Recording to {args.output_file}.avi")

    print("Processing started... Press Q to quit.")

    total_frames = 0
    frame_count = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: Camera frame not ready")
            time.sleep(0.01)
            continue

        total_frames += 1
        if total_frames % skip != 1:
            pass  # still show preview on skipped frames

        # Model inference
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb).resize(INF_SIZE)
            prediction = model.inference(image_pil)
            prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]))
        except Exception as e:
            print(f"Model inference failed: {e}")
            continue

        # Normalize depth
        pred_img = 255.0 * (prediction - np.min(prediction)) / max((np.max(prediction) - np.min(prediction)), 1e-5)
        pred_img = pred_img.astype(np.uint8)

        vis_obj = cv2.applyColorMap(pred_img, cmapy.cmap('viridis'))

        frame_hd = cv2.resize(frame, OUTPUT_SIZE)
        vis_obj_hd = cv2.resize(vis_obj, OUTPUT_SIZE)
        blended = cv2.addWeighted(vis_obj_hd, ALPHA_BLEND, frame_hd, 1 - ALPHA_BLEND, 0)

        # Write output
        if writer_obj:
            writer_obj.write(blended)

        # Preview
        if args.show:
            preview = cv2.resize(blended, PREVIEW_SIZE)
            cv2.imshow("Scene3D Live Preview", preview)

        # FPS monitor
        frame_count += 1
        t = time.time()
        if t - prev_time >= 1.0:
            print(f"FPS: {frame_count}")
            frame_count = 0
            prev_time = t

        # Quit preview with Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer_obj:
        writer_obj.release()
    cv2.destroyAllWindows()
    print("Completed.")


if __name__ == "__main__":
    main()
# %%

