#%%
#! /usr/bin/env python3
import cv2
import sys
import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.scene_seg_infer import SceneSegNetworkInfer

def find_freespace_edge(binary_mask):
    contours, _ = cv2.findContours(binary_mask,
                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    return cnt

def make_visualization_freespace(prediction, image):
    colour_mask = np.array(image)
    free_space_labels = np.where(prediction == 2)
    binary_mask = np.zeros(prediction.shape, dtype="uint8")
    binary_mask[free_space_labels[0], free_space_labels[1]] = 255
    edge_contour = find_freespace_edge(binary_mask)
    if edge_contour is not None and len(edge_contour) > 0:
        cv2.fillPoly(colour_mask, pts=[edge_contour], color=(28, 255, 145))
    colour_mask = cv2.cvtColor(colour_mask, cv2.COLOR_RGB2BGR)
    return colour_mask

def make_visualization(prediction):
    row, col = prediction.shape
    vis_predict_object = np.zeros((row, col, 3), dtype="uint8")
    vis_predict_object[:, :, 0] = 255
    vis_predict_object[:, :, 1] = 93
    vis_predict_object[:, :, 2] = 61
    foreground_labels = np.where(prediction == 1)
    vis_predict_object[foreground_labels[0], foreground_labels[1], 0] = 145
    vis_predict_object[foreground_labels[0], foreground_labels[1], 1] = 28
    vis_predict_object[foreground_labels[0], foreground_labels[1], 2] = 255
    return vis_predict_object

def main(): 
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", help="Path to PyTorch checkpoint file to load model")
    parser.add_argument("-i", "--input_image_filepath", dest="input_image_filepath", help="Path to input image for SceneSeg")
    parser.add_argument("-o", "--output_dir", dest="output_dir", help="Directory to save output images")
    args = parser.parse_args() 

    model_checkpoint_path = args.model_checkpoint_path
    model = SceneSegNetworkInfer(checkpoint_path=model_checkpoint_path)
    print('SceneSeg Model Loaded')

    alpha = 0.5

    # Read input image
    input_image_filepath = args.input_image_filepath
    frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"Error: Could not read image {input_image_filepath}")
        sys.exit(1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((640, 320))

    # Run inference and create visualizations
    prediction = model.inference(image_pil)
    vis_obj = make_visualization(prediction)
    vis_obj_freespace = make_visualization_freespace(prediction, image_pil)

    # Resize visualizations to 640x480
    image_vis_obj = cv2.addWeighted(cv2.resize(vis_obj, (640, 480)), alpha, cv2.resize(frame, (640, 480)), 1 - alpha, 0)
    image_vis_freespace = cv2.addWeighted(cv2.resize(vis_obj_freespace, (640, 480)), alpha, cv2.resize(frame, (640, 480)), 1 - alpha, 0)

    # Save directory from command line
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    input_filename = os.path.splitext(os.path.basename(input_image_filepath))[0]
    obj_save_path = os.path.join(save_dir, f"{input_filename}_objects.png")
    freespace_save_path = os.path.join(save_dir, f"{input_filename}_freespace.png")
    cv2.imwrite(obj_save_path, image_vis_obj)
    cv2.imwrite(freespace_save_path, image_vis_freespace)
    print(f"Saved object visualization to: {obj_save_path}")
    print(f"Saved freespace visualization to: {freespace_save_path}")

    # Display images
    cv2.imshow('Prediction Objects', image_vis_obj)
    cv2.imshow('Prediction Freespace', image_vis_freespace)

    # Wait for 'q' key to close windows
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
# %%

