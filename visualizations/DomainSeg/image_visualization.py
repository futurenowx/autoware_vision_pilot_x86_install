import cv2
import sys
import os
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from time import sleep

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.domain_seg_infer import DomainSegNetworkInfer

def make_visualization(prediction):
    prediction = np.squeeze(prediction)  # Remove singleton dimension
    if prediction.ndim != 2:
        raise ValueError(f"Expected 2D prediction, got {prediction.shape}")
    
    row, col = prediction.shape
    vis_predict_object = np.zeros((row, col, 3), dtype="uint8")
    
    # Background
    vis_predict_object[:, :, 0] = 255
    vis_predict_object[:, :, 1] = 93
    vis_predict_object[:, :, 2] = 61
    
    # Foreground
    fg = np.where(prediction == 1.0)
    vis_predict_object[fg[0], fg[1], 0] = 28
    vis_predict_object[fg[0], fg[1], 1] = 148
    vis_predict_object[fg[0], fg[1], 2] = 255
    
    return vis_predict_object

def main():
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", required=True)
    parser.add_argument("-i", "--input_image_filepath", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    args = parser.parse_args()
    
    # Load model
    model = DomainSegNetworkInfer(checkpoint_path=os.path.expanduser(args.model_checkpoint_path))
    print("DomainSeg Model Loaded")
    
    # Read image
    input_path = os.path.abspath(os.path.expanduser(args.input_image_filepath))
    frame = cv2.imread(input_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_pil = image_pil.resize((640, 320))
    
    # Inference
    prediction = model.inference(image_pil)
    vis_obj = make_visualization(prediction)
    
    # Resize to match frame for overlay
    vis_obj_overlay = cv2.resize(vis_obj, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create overlay
    alpha = 0.5
    overlay = cv2.addWeighted(vis_obj_overlay, alpha, frame, 1 - alpha, 0)
    
    # Convert to uint8 for saving
    overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Save overlay at 640x480
    save_overlay = cv2.resize(overlay_uint8, (640, 480), interpolation=cv2.INTER_NEAREST)
    
    output_folder = os.path.abspath(os.path.expanduser(args.output_folder))
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.basename(input_path)
    save_path = os.path.join(output_folder, filename)
    
    if not cv2.imwrite(save_path, save_overlay):
        raise RuntimeError(f"Failed to save image to {save_path}")
    
    print(f"Visualization saved to: {save_path}")
    
    # Display overlay window
    cv2.imshow("Prediction Overlay", overlay_uint8)
    print("Press Q to quit")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        sleep(0.01)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

