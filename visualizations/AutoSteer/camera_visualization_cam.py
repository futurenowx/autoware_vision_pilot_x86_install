# %%
# AutoSteer + EgoLanes (Camera Input, Live Preview) — FIXED

import sys
import cv2
import numpy as np
import os
from PIL import Image
from argparse import ArgumentParser
from datetime import datetime

# -------------------------
# Imports
# -------------------------
sys.path.append('../..')
from Models.inference.auto_steer_infer import AutoSpeedNetworkInfer
from inference.ego_lanes_infer import EgoLanesNetworkInfer

# -------------------------
# Constants
# -------------------------
OUTPUT_SIZE = (1280, 720)
PREVIEW_SIZE = (640, 320)
INF_SIZE = (640, 320)

# -------------------------
# Utility functions
# -------------------------
def rotate_wheel(wheel_img, angle_deg):
    h, w = wheel_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        wheel_img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )


def overlay_on_top(base_img, rotated_wheel_img, frame_time, steering_angle):
    H, W = base_img.shape[:2]
    oh, ow = rotated_wheel_img.shape[:2]
    x = W - ow - 60
    y = 20

    image = base_img.copy()

    def alpha_blend(dst, src, x, y):
        alpha = src[:, :, 3] / 255.0
        for c in range(3):
            dst[y:y+src.shape[0], x:x+src.shape[1], c] = (
                src[:, :, c] * alpha +
                dst[y:y+src.shape[0], x:x+src.shape[1], c] * (1 - alpha)
            )

    alpha_blend(image, rotated_wheel_img, x, y)

    cv2.putText(image, frame_time, (x - 60, y + oh + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(image, f"{steering_angle:.2f} deg", (x - 60, y + oh + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image


def overlay_egolanes(frame, lane_mask, threshold=0.3):
    overlay = frame.copy()
    H, W = frame.shape[:2]

    if lane_mask.ndim == 3:
        colors = [
            (255, 0, 255),
            (255, 0, 0),
            (0, 255, 0),
            (128, 0, 128)
        ]
        for i in range(lane_mask.shape[0]):
            lm = cv2.resize(lane_mask[i], (W, H), interpolation=cv2.INTER_NEAREST)
            overlay[lm >= threshold] = colors[i % len(colors)]
    else:
        lm = cv2.resize(lane_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        overlay[lm >= threshold] = (0, 255, 0)

    return cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)


# -------------------------
# Main
# -------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("-e", "--egolanes_checkpoint_path", required=True)
    parser.add_argument("-a", "--autosteer_checkpoint_path", required=True)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("-o", "--output_video_path", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    # -------------------------
    # Expand paths
    # -------------------------
    args.egolanes_checkpoint_path = os.path.expanduser(args.egolanes_checkpoint_path)
    args.autosteer_checkpoint_path = os.path.expanduser(args.autosteer_checkpoint_path)
    if args.output_video_path:
        args.output_video_path = os.path.expanduser(args.output_video_path)

    print("Loading models...")
    autosteer_model = AutoSpeedNetworkInfer(
        egolanes_checkpoint_path=args.egolanes_checkpoint_path,
        autosteer_checkpoint_path=args.autosteer_checkpoint_path
    )
    egolanes_model = EgoLanesNetworkInfer(args.egolanes_checkpoint_path)
    print("Models loaded")

    # -------------------------
    # Camera
    # -------------------------
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError("❌ Could not open camera")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # -------------------------
    # Video writer (FIXED)
    # -------------------------
    writer = None
    if args.output_video_path:
        # Ensure .avi extension
        if not args.output_video_path.lower().endswith(".avi"):
            args.output_video_path += ".avi"

        os.makedirs(os.path.dirname(args.output_video_path), exist_ok=True)

        writer = cv2.VideoWriter(
            args.output_video_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            OUTPUT_SIZE
        )

        if not writer.isOpened():
            raise RuntimeError("❌ Failed to open VideoWriter")

        print(f"Recording to {args.output_video_path}")

    # -------------------------
    # Load wheel image
    # -------------------------
    media_path = os.path.join(
        os.path.dirname(__file__),
        "../../../Media/wheel_green.png"
    )

    if not os.path.exists(media_path):
        raise FileNotFoundError(f"Wheel image not found: {media_path}")

    wheel = cv2.resize(cv2.imread(media_path, cv2.IMREAD_UNCHANGED), None, fx=0.8, fy=0.8)

    print("Starting camera inference (press Q to quit)")

    # -------------------------
    # Loop
    # -------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(INF_SIZE)

        steering_angle = autosteer_model.inference(pil)
        lane_mask = egolanes_model.inference(np.array(pil))

        frame = cv2.resize(frame, OUTPUT_SIZE)
        frame = overlay_egolanes(frame, lane_mask)

        rotated_wheel = rotate_wheel(wheel, steering_angle)
        timestamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        frame = overlay_on_top(frame, rotated_wheel, timestamp, steering_angle)

        if args.show:
            cv2.imshow("AutoSteer + EgoLanes", cv2.resize(frame, PREVIEW_SIZE))

        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
# %%

