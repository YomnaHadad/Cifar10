import keras
from keras.models import Sequential
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Input
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model2 = keras.Sequential(
    [
        Input(shape=(32,32,3)),
        Conv2D(32, kernel_size=(3,3), activation='relu'),
        BatchNormalization(),
        Conv2D(32, kernel_size=(3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.3),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.5),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ]
)
def plot_loss(history):
  plt.figure(figsize=[6,4])
  plt.plot(history.history['loss'], 'black')
  plt.plot(history.history['val_loss'], 'green')
  plt.legend(['Training Loss', 'Validation Loss'])
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Loss Curves')

def plot_accuracy(history):
  plt.figure(figsize=[6,4])
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.legend(['Training Accuracy', 'Validation Accuracy'])
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.title('Accuracy Curves')

if __name__ == "__main__" :
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  print(f"x_train shape: {x_train.shape}")
  print(f"y_train shape: {y_train.shape}")
  print(f"x_test shape: {x_test.shape}")
  print(f"y_test shape: {y_test.shape}")

  x_train = x_train / 255.
  x_test = x_test / 255.

  model2.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
  print(model2.summary())

  history = model2.fit(x_train, y_train, epochs= 100, batch_size=32, verbose=2, validation_split=0.1, callbacks=callbacks)

  ### Testing
  test_loss, test_accuracy = model2.evaluate(x_test, y_test)

  plot_loss(history)
  plot_accuracy(history)
