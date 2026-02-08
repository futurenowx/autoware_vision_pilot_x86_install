from argparse import ArgumentParser
import os
import cv2
import sys
from PIL import Image
from inference.auto_speed_infer import AutoSpeedNetworkInfer

import warnings
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid"
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


color_map = {
    1: (0, 0, 255),    # red
    2: (0, 255, 255),  # yellow
    3: (255, 255, 0)   # cyan
}


def make_visualization(prediction, input_image_filepath, output_path=None):
    img_cv = cv2.imread(input_image_filepath)

    for pred in prediction:
        x1, y1, x2, y2, conf, cls = pred
        color = color_map.get(int(cls), (255, 255, 255))

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img_cv)
        print(f"[OK] Saved output to: {output_path}")
    else:
        # Optional: only show if no output path provided
        cv2.imshow("Prediction Objects", img_cv)
        cv2.waitKey(1)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-p", "--model_checkpoint_path",
        required=True,
        help="Path to PyTorch checkpoint"
    )
    parser.add_argument(
        "-i", "--input_image_filepath",
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output image file path (e.g. output/result.png)"
    )

    args = parser.parse_args()

    model = AutoSpeedNetworkInfer(args.model_checkpoint_path)
    img = Image.open(args.input_image_filepath).convert("RGB")

    prediction = model.inference(img)
    make_visualization(prediction, args.input_image_filepath, args.output)

