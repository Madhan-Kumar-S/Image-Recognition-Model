import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def load_data():
    X_train = np.load('X_train.npy')
    X_val = np.load('X_val.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')

    y_train = to_categorical(y_train)  # One-hot encode labels
    y_val = to_categorical(y_val)

    return X_train, X_val, y_train, y_val


def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))  # 2 classes: cat and dog

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model():
    X_train, X_val, y_train, y_val = load_data()

    model = build_model()

    history = model.fit(X_train, y_train,
                        epochs=10,  # Adjust the number of epochs based on your needs
                        batch_size=32,
                        validation_data=(X_val, y_val))

    model.save('cat_dog_model.h5')  # Save the trained model

    # Save accuracy history
    np.save('training_history.npy', history.history)

    return history


if __name__ == "__main__":
    history = train_model()

    # Print training and validation accuracy
    print(f"Training accuracy: {history.history['accuracy'][-1]}")
    print(f"Validation accuracy: {history.history['val_accuracy'][-1]}")

    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.savefig('accuracy_plot.png')
    plt.show()
