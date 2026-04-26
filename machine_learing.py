import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Path to dataset (organized as dataset/class_name/*.jpg)
DATASET_PATH = "dataset/"

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram of Oriented Gradients (HOG)
    hog = cv2.HOGDescriptor()
    h = hog.compute(gray)

    return h.flatten()

X, y = [], []
classes = os.listdir(DATASET_PATH)

for idx, cls in enumerate(classes):
    cls_path = os.path.join(DATASET_PATH, cls)
    for file in os.listdir(cls_path):
        fpath = os.path.join(cls_path, file)
        features = extract_features(fpath)
        X.append(features)
        y.append(idx)

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=classes))
