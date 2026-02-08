#%%
import cv2
import sys
import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser

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

def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", required=True)
    parser.add_argument("-i", "--video_filepath", required=True)
    parser.add_argument("-s", "--show_scale", type=float, default=1.0)
    parser.add_argument("-o", "--output_file", required=True)
    parser.add_argument("--skip_frames", type=int, default=0)
    args = parser.parse_args()

    model = SceneSegNetworkInfer(checkpoint_path=args.model_checkpoint_path)
    print('SceneSeg Model Loaded')

    cap = cv2.VideoCapture(args.video_filepath)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    output_obj_path = args.output_file + '.avi'
    output_free_path = args.output_file + '_freespace.avi'

    # Adjust output FPS for skipped frames
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    skip = args.skip_frames + 1
    fps_output = fps_input / skip

    writer_obj = cv2.VideoWriter(output_obj_path, cv2.VideoWriter_fourcc(*"MJPG"), fps_output, (640, 400))
    writer_freespace = cv2.VideoWriter(output_free_path, cv2.VideoWriter_fourcc(*"MJPG"), fps_output, (640, 400))

    alpha = 0.5
    print('Processing started')

    frame_count = 0
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        if total_frames % skip != 1:
            continue  # skip frames

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image).resize((640, 320))

        prediction = model.inference(image_pil)
        vis_obj = make_visualization(prediction)
        vis_obj_freespace = make_visualization_freespace(prediction, image_pil)

        frame_resized = cv2.resize(frame, (640, 400))
        vis_obj_resized = cv2.resize(vis_obj, (640, 400))
        vis_obj_freespace_resized = cv2.resize(vis_obj_freespace, (640, 400))

        image_vis_obj = cv2.addWeighted(vis_obj_resized, alpha, frame_resized, 1 - alpha, 0)
        image_vis_freespace = cv2.addWeighted(vis_obj_freespace_resized, alpha, frame_resized, 1 - alpha, 0)

        # Write to output video
        writer_obj.write(image_vis_obj)
        writer_freespace.write(image_vis_freespace)

        # Live preview scaled
        if args.show_scale != 1.0:
            w = int(image_vis_obj.shape[1] * args.show_scale)
            h = int(image_vis_obj.shape[0] * args.show_scale)
            disp_obj = cv2.resize(image_vis_obj, (w, h))
            disp_free = cv2.resize(image_vis_freespace, (w, h))
        else:
            disp_obj = image_vis_obj
            disp_free = image_vis_freespace

        cv2.imshow('Prediction Objects', disp_obj)
        cv2.imshow('Prediction Freespace', disp_free)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer_obj.release()
    writer_freespace.release()
    cv2.destroyAllWindows()
    print('Completed')

if __name__ == '__main__':
    main()

