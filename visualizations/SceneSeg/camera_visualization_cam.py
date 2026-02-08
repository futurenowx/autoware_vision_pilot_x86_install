#%%
import cv2
import sys
import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.scene_seg_infer import SceneSegNetworkInfer

def find_freespace_edge(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        return max(contours, key=lambda x: cv2.contourArea(x))
    return None

def make_visualization_freespace(prediction, image):
    colour_mask = np.array(image)
    free_space_labels = np.where(prediction == 2)
    row, col = prediction.shape
    binary_mask = np.zeros((row, col), dtype="uint8")
    binary_mask[free_space_labels[0], free_space_labels[1]] = 255
    edge_contour = find_freespace_edge(binary_mask)
    if edge_contour is not None and edge_contour.size > 0:
        cv2.fillPoly(colour_mask, pts=[edge_contour], color=(28, 255, 145))
    colour_mask = cv2.cvtColor(colour_mask, cv2.COLOR_RGB2BGR)
    return colour_mask

def make_visualization(prediction):
    row, col = prediction.shape
    vis = np.zeros((row, col, 3), dtype="uint8")
    vis[:, :, 0] = 255
    vis[:, :, 1] = 93
    vis[:, :, 2] = 61
    fg = np.where(prediction == 1)
    vis[fg[0], fg[1], 0] = 145
    vis[fg[0], fg[1], 1] = 28
    vis[fg[0], fg[1], 2] = 255
    return vis

# -------------------------
OUTPUT_SIZE = (640, 400)
PREVIEW_SIZE = (640, 320)
ALPHA_BLEND = 0.5
# -------------------------

def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", required=True)
    parser.add_argument("--camera_id", type=int, default=6)
    parser.add_argument("-o", "--output_file", default=None)
    parser.add_argument("--skip_frames", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    # Load model
    model = SceneSegNetworkInfer(checkpoint_path=args.model_checkpoint_path)
    print("SceneSeg Model Loaded")

    # Open camera
    print(f"Opening camera ID {args.camera_id}...")
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera_id}")
        return

    # Warm up camera
    for _ in range(20):
        cap.read()

    fps_input = cap.get(cv2.CAP_PROP_FPS) or 15
    skip = args.skip_frames + 1
    fps_output = fps_input / skip

    # Optional writers
    writer_obj = None
    writer_free = None
    if args.output_file:
        writer_obj = cv2.VideoWriter(args.output_file + ".avi", cv2.VideoWriter_fourcc(*"MJPG"), fps_output, OUTPUT_SIZE)
        writer_free = cv2.VideoWriter(args.output_file + "_freespace.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps_output, OUTPUT_SIZE)
        print(f"Recording to {args.output_file}.avi and {args.output_file}_freespace.avi")

    print("Processing started... Press Q to quit.")

    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        total_frames += 1
        if total_frames % skip != 1:
            pass  # still show preview on skipped frames

        # Convert to RGB PIL for model
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb).resize((640, 320))

        prediction = model.inference(image_pil)
        vis_obj = make_visualization(prediction)
        vis_free = make_visualization_freespace(prediction, image_pil)

        frame_resized = cv2.resize(frame, OUTPUT_SIZE)
        vis_obj_resized = cv2.resize(vis_obj, OUTPUT_SIZE)
        vis_free_resized = cv2.resize(vis_free, OUTPUT_SIZE)

        blended_obj = cv2.addWeighted(vis_obj_resized, ALPHA_BLEND, frame_resized, 1-ALPHA_BLEND, 0)
        blended_free = cv2.addWeighted(vis_free_resized, ALPHA_BLEND, frame_resized, 1-ALPHA_BLEND, 0)

        # Write outputs
        if writer_obj: writer_obj.write(blended_obj)
        if writer_free: writer_free.write(blended_free)

        # Live preview
        if args.show:
            cv2.imshow("SceneSeg Objects Preview", cv2.resize(blended_obj, PREVIEW_SIZE))
            cv2.imshow("SceneSeg Freespace Preview", cv2.resize(blended_free, PREVIEW_SIZE))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer_obj: writer_obj.release()
    if writer_free: writer_free.release()
    cv2.destroyAllWindows()
    print("Completed.")

if __name__ == "__main__":
    main()
# %%

