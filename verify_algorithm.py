import cv2
from src.core.detection import ForeignBodyDetector
from src.core.classification import ParticleClassifier

def main():
    img = cv2.imread("test_sample.jpg")
    if img is None:
        print("Failed to load image")
        return

    detector = ForeignBodyDetector()
    classifier = ParticleClassifier()

    # Run detection
    contours, processed = detector.detect_static(img, threshold=200, min_area=5, use_adaptive=False)

    print(f"Detected {len(contours)} objects.")

    for i, cnt in enumerate(contours):
        result = classifier.classify(cnt)
        print(f"Object {i+1}: {result}")

        # Visualize
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img, result['label'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite("verification_result.jpg", img)
    print("Result saved to verification_result.jpg")

if __name__ == "__main__":
    main()
