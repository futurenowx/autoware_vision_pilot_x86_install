#%%
import cv2
import sys
import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.domain_seg_infer import DomainSegNetworkInfer


def make_visualization(prediction):
    prediction = np.squeeze(prediction)  # ensure 2D
    row, col = prediction.shape
    vis_predict_object = np.zeros((row, col, 3), dtype="uint8")

    # Assign background colour
    vis_predict_object[:, :, 0] = 255
    vis_predict_object[:, :, 1] = 93
    vis_predict_object[:, :, 2] = 61

    # Get foreground object labels
    foreground_labels = np.where(prediction == 1.0)

    # Assign foreground objects colour
    vis_predict_object[foreground_labels[0], foreground_labels[1], 0] = 28
    vis_predict_object[foreground_labels[0], foreground_labels[1], 1] = 148
    vis_predict_object[foreground_labels[0], foreground_labels[1], 2] = 255

    return vis_predict_object


def main(): 
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", required=True, help="path to pytorch checkpoint file")
    parser.add_argument("-i", "--video_filepath", required=True, help="path to input video")
    parser.add_argument("-o", "--output_file", required=True, help="path to output video file")
    parser.add_argument("--skip_frames", type=int, default=0, help="skip every N frames (0 = no skip)")
    parser.add_argument("--no_vis", action="store_true", help="disable live preview")
    args = parser.parse_args() 

    # Load model
    model = DomainSegNetworkInfer(checkpoint_path=os.path.expanduser(args.model_checkpoint_path))
    print('DomainSeg Model Loaded')

    # Open video
    video_path = os.path.expanduser(args.video_filepath)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        return

    # Video writer
    output_filepath_obj = os.path.expanduser(args.output_file) + '.avi'
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer_obj = cv2.VideoWriter(
        output_filepath_obj,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps / (args.skip_frames + 1),  # adjust fps for skipped frames
        (frame_width, frame_height)
    )

    alpha = 0.5
    print('Processing started')

    total_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('End of video reached')
            break

        total_frames += 1
        if args.skip_frames > 0 and total_frames % (args.skip_frames + 1) != 0:
            continue

        # Convert frame for inference
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb).resize((640, 320))

        # Run inference
        prediction = model.inference(image_pil)
        prediction = np.squeeze(prediction)  # ensure 2D

        # Visualization
        vis_obj = make_visualization(prediction)
        vis_obj_resized = cv2.resize(vis_obj, (frame.shape[1], frame.shape[0]))

        # Blend original frame with prediction
        image_vis_obj = cv2.addWeighted(vis_obj_resized, alpha, frame, 1 - alpha, 0)

        # Save frame
        writer_obj.write(image_vis_obj)

        # Live preview
        if not args.no_vis:
            preview = cv2.resize(image_vis_obj, (640, 480))
            cv2.imshow('DomainSeg Prediction', preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer_obj.release()
    cv2.destroyAllWindows()
    print('Completed')


if __name__ == '__main__':
    main()
# %%

