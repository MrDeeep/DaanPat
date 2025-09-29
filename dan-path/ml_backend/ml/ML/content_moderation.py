from nudenet import NudeClassifier, NudeDetector
import os


# ======================
# 1. Initialize Models
# ======================
# Classifier -> predicts whether image is SAFE / UNSAFE
classifier = NudeClassifier()

# Detector -> detects & localizes NSFW regions (breast, buttocks, etc.)
detector = NudeDetector()

# ======================
# 2. Single Image Classification
# ======================
def classify_image(image_path):
    result = classifier.classify(image_path)
    print(f"\n[Classification] {image_path}: {result}")
    return result

# ======================
# 3. Batch Classification
# ======================
def classify_folder(folder_path):
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    results = classifier.classify(images)
    print("\n[Batch Results]:")
    for img, res in results.items():
        print(f"{img}: {res}")
    return results

# ======================
# 4. Detect NSFW Regions
# ======================
def detect_nsfw_regions(image_path):
    detections = detector.detect(image_path)
    print(f"\n[Detection] {image_path}:")
    for d in detections:
        print(d)
    return detections

# ======================
# 5. Example Usage
# ======================
if __name__ == "__main__":
    test_image = "test.jpg"       # change this to your test image path
    test_folder = "test_images"   # change this to your folder path

    if os.path.exists(test_image):
        classify_image(test_image)
        detect_nsfw_regions(test_image)

    if os.path.exists(test_folder):
        classify_folder(test_folder)
