import cv2
import argparse
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.detection import ForeignBodyDetector
from src.core.classification import ParticleClassifier

def main():
    parser = argparse.ArgumentParser(description="Run Vial Inspection on a single image.")
    parser.add_argument("image_path", help="Path to the image file.")
    parser.add_argument("--output", default="result.jpg", help="Path to save the result image.")
    parser.add_argument("--thresh", type=int, default=100, help="Binary threshold value.")
    parser.add_argument("--min_area", type=int, default=10, help="Minimum area for particles.")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: File {args.image_path} not found.")
        return

    img = cv2.imread(args.image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    detector = ForeignBodyDetector()
    classifier = ParticleClassifier()

    print(f"Processing {args.image_path}...")
    contours, processed_img = detector.detect_static(img, threshold=args.thresh, min_area=args.min_area)

    print(f"Detected {len(contours)} objects.")

    result_img = img.copy()
    if len(result_img.shape) == 2:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)

    for i, cnt in enumerate(contours):
        res = classifier.classify(cnt)
        print(f"  #{i+1}: {res['label']} (Conf: {res.get('confidence', 1.0):.2f}, Area: {res['area']})")

        # Draw
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label_text = f"{res['label']}"
        cv2.putText(result_img, label_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(args.output, result_img)
    print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()
