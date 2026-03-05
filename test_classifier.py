import cv2
import numpy as np
from src.core.classification import DeepLearningClassifier

classifier = DeepLearningClassifier()
ok, err = classifier.load_model("classification_model.onnx")
print("Model load:", ok, err)
print("Labels:", classifier.labels)

# Create a dummy image and contour
dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.circle(dummy_img, (50, 50), 20, (255, 255, 255), -1)
gray = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    res = classifier.classify_batch(contours, dummy_img)
    print("Result:", res)
else:
    print("No contours found")
