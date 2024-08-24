import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


def load_images_from_folder(folder):
    images = []
    labels = []
    label_map = {'cats': 0, 'dogs': 1}  # Map subdirectory names to labels

    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
                    img_path = os.path.join(subdir_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (128, 128))  # Resize images for consistency
                        images.append(img)
                        labels.append(label_map.get(subdir))  # Get label from subdir name
                    else:
                        print(f"Warning: Unable to read image {filename} in {subdir_path}")
                else:
                    print(f"Warning: Skipping non-image file {filename} in {subdir_path}")

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def preprocess_data(X, y):
    if len(X) == 0 or len(y) == 0:
        raise ValueError("The dataset is empty. Check image loading.")
    X = X / 255.0  # Normalize pixel values
    return train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    train_folder ='' # Update this path to your actual path
    X, y = load_images_from_folder(train_folder)

    print(f"Loaded {len(X)} images")
    print(f"Labels: {np.unique(y, return_counts=True)}")

    X_train, X_val, y_train, y_val = preprocess_data(X, y)

    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
