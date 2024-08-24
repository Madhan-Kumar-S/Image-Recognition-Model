import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # Resize to match model input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def predict(img_path):
    model = load_model('cat_dog_model.h5')
    img = preprocess_image(img_path)
    predictions = model.predict(img)
    class_idx = np.argmax(predictions[0])
    return 'cat' if class_idx == 0 else 'dog'


def plot_accuracy():
    history = np.load('training_history.npy', allow_pickle=True).item()

    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.show()


def print_model_accuracy():
    history = np.load('training_history.npy', allow_pickle=True).item()

    # Assuming the history dictionary has 'accuracy' and 'val_accuracy' lists
    final_train_accuracy = history['accuracy'][-1] * 100
    final_val_accuracy = history['val_accuracy'][-1] * 100

    print(f"Final Model Accuracy: {final_train_accuracy:.2f}%")
    print(f"Final Validation Accuracy: {final_val_accuracy:.2f}%")


if __name__ == "__main__":

    img_path = ''#add image path
    result = predict(img_path)
    print(f"The image is classified as: {result}")

    # Display the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    plt.imshow(img)
    plt.title(f'Predicted: {result}')
    plt.axis('off')  # Hide axes
    plt.show()

    # Print model accuracy
    print_model_accuracy()

    # Plot accuracy graph
    plot_accuracy()
