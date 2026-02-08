#!/usr/bin/env python3
import cv2
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import os

# Add repo root to Python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference.auto_speed_infer import AutoSpeedNetworkInfer

color_map = {
    1: (0, 0, 255),    # red
    2: (0, 255, 255),  # yellow
    3: (255, 255, 0)   # cyan
}

# Preview window size
PREVIEW_SIZE = (640, 320)
FRAME_ORI_SIZE = None  # Will be set dynamically from camera

def make_visualization(prediction, image):
    for pred in prediction:
        x1, y1, x2, y2, conf, cls = pred
        color = color_map.get(int(cls), (255, 255, 255))
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", required=True,
                        help="Path to PyTorch checkpoint (.pt) or ONNX model (.onnx)")
    parser.add_argument("--camera_id", type=int, default=6,
                        help="Camera device index (default: 0)")
    parser.add_argument("-o", "--output_file", default=None,
                        help="Optional output video path")
    parser.add_argument("-s", "--show", action="store_true",
                        help="Flag to show live preview")
    args = parser.parse_args()

    # Load model
    model_path = args.model_checkpoint_path
    if model_path.endswith('.onnx'):
        from Models.inference.video_onnx_infer import AutoSpeedONNXInfer
        print('Loading ONNX model...')
        model = AutoSpeedONNXInfer(model_path)
    else:
        print('Loading PyTorch model...')
        model = AutoSpeedNetworkInfer(model_path)

    print("Model loaded successfully")

    # -------------------------
    # Open camera
    # -------------------------
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print("Error opening camera.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback for webcams

    global FRAME_ORI_SIZE
    FRAME_ORI_SIZE = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # -------------------------
    # Optional video writer
    # -------------------------
    writer = None
    if args.output_file is not None:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        writer = cv2.VideoWriter(
            args.output_file,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            FRAME_ORI_SIZE
        )
        print(f"Recording output to: {args.output_file}")

    last_preview = None
    print("Starting camera inference (press Q to quit)")

    # -------------------------
    # Camera loop
    # -------------------------
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            # Inference
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            prediction = model.inference(image_pil)

            # Visualization
            vis_frame = make_visualization(prediction, frame.copy())

            # Write to video if enabled
            if writer is not None:
                writer.write(vis_frame)

            # Preview window
            if args.show:
                preview = cv2.resize(vis_frame, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
                last_preview = preview
                cv2.imshow("Object Detection Preview", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Stopped by user.")
                break

    except KeyboardInterrupt:
        print("Interrupted by Ctrl+C")

    # -------------------------
    # Cleanup
    # -------------------------
    cap.release()
    if writer is not None:
        writer.release()

    # Endless last-frame preview
    if args.show and last_preview is not None:
        print("Entering endless preview mode (press Q to exit)")
        while True:
            cv2.imshow("Object Detection Preview", last_preview)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()

