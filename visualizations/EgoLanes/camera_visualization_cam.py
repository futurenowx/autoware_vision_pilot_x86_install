import sys
import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser

sys.path.append("../..")

from inference.ego_lanes_infer import EgoLanesNetworkInfer
from image_visualization import make_visualization

FRAME_INF_SIZE = (640, 320)
FRAME_ORI_SIZE = (720, 360)
PREVIEW_SIZE = (640, 360)


def main():

    parser = ArgumentParser()

    parser.add_argument(
        "-p",
        "--model_checkpoint_path",
        required=True,
        help="Path to EgoLanes PyTorch checkpoint"
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=6,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "-o",
        "--output_video_path",
        default=None,
        help="Optional output video path"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live visualization"
    )

    args = parser.parse_args()

    # -------------------------
    # Load model
    # -------------------------
    print("Loading EgoLanes model...")
    model = EgoLanesNetworkInfer(
        checkpoint_path=args.model_checkpoint_path
    )
    print("EgoLanes model successfully loaded!")

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

    # -------------------------
    # Optional video writer
    # -------------------------
    writer = None
    if args.output_video_path is not None:
        writer = cv2.VideoWriter(
            args.output_video_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            FRAME_ORI_SIZE
        )
        print(f"Recording output to: {args.output_video_path}")

    last_preview = None

    print("Starting camera inference (press Q or S to quit)")

    # -------------------------
    # Camera loop (endless)
    # -------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        # Resize for inference
        image = Image.fromarray(frame)
        image = image.resize(FRAME_INF_SIZE)
        image = np.array(image)

        # Inference
        prediction = model.inference(image)

        # Visualization
        vis_image = make_visualization(image.copy(), prediction)
        vis_image = np.array(vis_image)
        vis_image = cv2.resize(vis_image, FRAME_ORI_SIZE)

        # Write to video if enabled
        if writer is not None:
            writer.write(vis_image)

        # Preview
        if args.show:
            preview = cv2.resize(vis_image, PREVIEW_SIZE)
            last_preview = preview
            cv2.imshow("EgoLanes Visualization", preview)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("s")):
            print("Stopped by user.")
            break

    # -------------------------
    # Cleanup
    # -------------------------
    cap.release()
    if writer is not None:
        writer.release()

    # Endless last-frame preview
    if args.show and last_preview is not None:
        print("Entering endless preview mode (press Q or S to exit)")
        while True:
            cv2.imshow("EgoLanes Visualization", last_preview)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("s")):
                break

    cv2.destroyAllWindows()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()

