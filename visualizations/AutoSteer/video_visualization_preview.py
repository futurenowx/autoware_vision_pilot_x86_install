# %%
# AutoSteer + EgoLanes Visualization (all lanes, self-contained)

import sys
import cv2
import json
import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from datetime import datetime, timedelta

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


def overlay_on_top(base_img, rotated_wheel_img, frame_time, steering_angle, rotated_gt_wheel_img=None):
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

    if rotated_gt_wheel_img is not None:
        alpha_blend(image, rotated_gt_wheel_img, x - 148, y)

    cv2.putText(image, frame_time, (x - 60, y + oh + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(image, f"{steering_angle:.2f} deg", (x - 60, y + oh + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image


def load_ground_truth(gt_file_path):
    with open(gt_file_path, 'r') as f:
        return json.load(f)


def overlay_egolanes(frame, lane_mask, threshold=0.3):
    """
    Draw all EgoLanes with custom colors
    """
    overlay = frame.copy()
    H, W = frame.shape[:2]

    if lane_mask.ndim == 3:
        colors = [
            (255, 0, 255),  # purple
            (255, 0, 0),    # blue
            (0, 255, 0),    # green
            (128, 0, 128)   # dark purple
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
    parser.add_argument("-i", "--video_filepath", required=True)
    parser.add_argument("-o", "--output_file", required=True)
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    parser.add_argument("-g", "--ground_truth")
    args = parser.parse_args()

    print("Loading models...")
    autosteer_model = AutoSpeedNetworkInfer(
        egolanes_checkpoint_path=args.egolanes_checkpoint_path,
        autosteer_checkpoint_path=args.autosteer_checkpoint_path
    )
    egolanes_model = EgoLanesNetworkInfer(args.egolanes_checkpoint_path)
    print("Models loaded")

    cap = cv2.VideoCapture(args.video_filepath)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps < 1:
        print("[WARN] FPS not detected, fallback to 30 FPS")
        fps = 30.0

    writer = cv2.VideoWriter(
        args.output_file + ".avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        OUTPUT_SIZE
    )

    # -------------------------
    # Load wheels (relative path)
    # -------------------------
    media_path = os.path.join(
        os.path.dirname(__file__),
        "../../../Media/wheel_green.png"
    )

    if not os.path.exists(media_path):
        raise FileNotFoundError(f"Wheel image not found: {media_path}")

    wheel = cv2.resize(cv2.imread(media_path, cv2.IMREAD_UNCHANGED), None, fx=0.8, fy=0.8)
    gt_wheel = wheel.copy()


    gt = load_ground_truth(args.ground_truth) if args.ground_truth else None
    start_datetime = datetime.now()
    frame_idx = 0

    # -------------------------
    # Processing loop
    # -------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize(INF_SIZE)

        steering_angle = autosteer_model.inference(pil)
        lane_mask = egolanes_model.inference(np.array(pil))

        frame = cv2.resize(frame, OUTPUT_SIZE)
        frame = overlay_egolanes(frame, lane_mask)

        rotated_wheel = rotate_wheel(wheel, steering_angle)
        rotated_gt = None

        if gt:
            gt_angle = gt["frames"][frame_idx]["steering_angle_corrected"]
            rotated_gt = rotate_wheel(gt_wheel, gt_angle)

        timestamp = (start_datetime + timedelta(seconds=frame_idx / fps)).strftime(
            "%m/%d/%Y %H:%M:%S"
        )

        frame = overlay_on_top(
            frame,
            rotated_wheel,
            timestamp,
            steering_angle,
            rotated_gt
        )

        # -------------------------
        # Preview window (640x320)
        # -------------------------
        if args.show:
            preview = cv2.resize(frame, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imshow("AutoSteer + EgoLanes (Preview)", preview)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Completed. Output saved to {args.output_file}.avi")


if __name__ == "__main__":
    main()
# %%

