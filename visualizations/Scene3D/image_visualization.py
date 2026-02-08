#%%
#! /usr/bin/env python3
import cv2
import sys
import os
import numpy as np
from argparse import ArgumentParser
import cmapy
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.scene_3d_infer import Scene3DNetworkInfer

def main(): 
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", help="Path to PyTorch checkpoint file to load model")
    parser.add_argument("-i", "--input_image_filepath", dest="input_image_filepath", help="Path to input image for Scene3D")
    parser.add_argument("-o", "--output_dir", dest="output_dir", help="Directory to save output image")
    args = parser.parse_args() 

    # Load model
    model_checkpoint_path = args.model_checkpoint_path
    model = Scene3DNetworkInfer(checkpoint_path=model_checkpoint_path)
    print("Scene3D Model Loaded")

    # Read input image
    input_image_filepath = args.input_image_filepath
    frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"Error: Could not read image {input_image_filepath}")
        sys.exit(1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((640, 320))

    # Run inference
    prediction = model.inference(image_pil)

    # Resize prediction to original frame size first
    prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]))

    # Transparency factor
    alpha = 0.97

    # Create visualization
    prediction_image = 255.0 * ((prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction)))
    prediction_image = prediction_image.astype(np.uint8)
    prediction_image = cv2.applyColorMap(prediction_image, cmapy.cmap('viridis'))

    # Resize both to 640x480 for saving
    frame_resized = cv2.resize(frame, (640, 480))
    prediction_image_resized = cv2.resize(prediction_image, (640, 480))
    image_vis_obj = cv2.addWeighted(prediction_image_resized, alpha, frame_resized, 1 - alpha, 0)

    # Save directory from command line
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    input_filename = os.path.splitext(os.path.basename(input_image_filepath))[0]
    save_path = os.path.join(save_dir, f"{input_filename}_depth.png")
    cv2.imwrite(save_path, image_vis_obj)
    print(f"Saved depth visualization to: {save_path}")

    # Display depth map
    window_name = 'depth'
    cv2.imshow(window_name, image_vis_obj)

    # Wait for 'q' key to close
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
# %%

