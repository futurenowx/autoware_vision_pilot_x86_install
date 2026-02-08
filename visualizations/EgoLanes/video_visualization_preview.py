import sys
import cv2
import numpy as np
from tqdm import tqdm
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
        "-i",
        "--input_video_filepath",
        required=True,
        help="Path to input video"
    )
    parser.add_argument(
        "-o",
        "--output_video_path",
        required=True,
        help="Path to output visualization video"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live visualization while processing"
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
    # Open input video
    # -------------------------
    cap = cv2.VideoCapture(args.input_video_filepath)
    if not cap.isOpened():
        print("Error opening video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # -------------------------
    # Output writer
    # -------------------------
    writer = cv2.VideoWriter(
        args.output_video_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        FRAME_ORI_SIZE
    )

    last_preview = None

    # -------------------------
    # Process video
    # -------------------------
    for i in tqdm(
        range(frame_count),
        desc="Processing video frames",
        unit="frames",
        colour="green"
    ):
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(frame)
        image = image.resize(FRAME_INF_SIZE)
        image = np.array(image)

        prediction = model.inference(image)

        vis_image = make_visualization(image.copy(), prediction)
        vis_image = np.array(vis_image)
        vis_image = cv2.resize(vis_image, FRAME_ORI_SIZE)

        writer.write(vis_image)

        if args.show:
            preview = cv2.resize(vis_image, PREVIEW_SIZE)
            last_preview = preview
            cv2.imshow("EgoLanes Visualization", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("s")):
                print("Stopped by user.")
                break

    # -------------------------
    # Release video resources
    # -------------------------
    cap.release()
    writer.release()

    print(f"Visualization video saved to: {args.output_video_path}")

    # -------------------------
    # Endless preview loop
    # -------------------------
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

