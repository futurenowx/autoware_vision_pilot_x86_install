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
    parser.add_argument("-i", "--video_filepath", required=True,
                        help="Path to input video")
    parser.add_argument("-o", "--output_file", required=True,
                        help="Path to output video file (include filename, e.g. output/result.avi)")
    parser.add_argument("-s", "--show", action="store_true",
                        help="Flag to show frame-by-frame visualization")
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

    # Ensure output folder exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    cap = cv2.VideoCapture(args.video_filepath)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        args.output_file,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height)
    )

    print("Processing started...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            prediction = model.inference(image_pil)
            vis_frame = make_visualization(prediction, frame.copy())

            # -------------------------
            # Preview window (640x320)
            # -------------------------
            if args.show:
                preview = cv2.resize(vis_frame, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
                cv2.imshow("Object Detection Preview", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Interrupted by user")
                    break

            writer.write(vis_frame)

    except KeyboardInterrupt:
        print("Interrupted by Ctrl+C")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Video saved to {args.output_file}")


if __name__ == "__main__":
    main()

