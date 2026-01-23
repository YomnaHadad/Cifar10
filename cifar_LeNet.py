import tensorflow
import keras
from keras.models import Sequential
from keras.datasets import cifar10
from keras.layers import Conv2D, AveragePooling2D, Flatten, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# LeNet Standard Architecture
model3 = keras.Sequential(
    [
        Input(shape=(32,32,3)),
        Conv2D(6, kernel_size=(5,5), activation='tanh', strides=1),
        AveragePooling2D(pool_size=(2,2), strides=2),

        Conv2D(16, (5,5), activation='tanh', strides=1),
        AveragePooling2D(pool_size=(2,2), strides=2),
        Conv2D(120, (5,5), activation='tanh', strides=1),


        Flatten(),
        Dense(84, activation='tanh'),
        Dense(10, activation='softmax')
    ]
)

if __name__ == "__main__" :
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  print(f"x_train shape: {x_train.shape}")
  print(f"y_train shape: {y_train.shape}")
  print(f"x_test shape: {x_test.shape}")
  print(f"y_test shape: {y_test.shape}")

  x_train = x_train / 255.
  x_test = x_test / 255.
  
  datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
  )
  datagen.fit(x_train)

  optimizer = keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
  )

  model3.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])
  print(model3.summary())

  history = model3.fit(x_train, y_train, epochs= 50, batch_size=32, verbose=2, validation_split=0.1)

  ### Testing
  test_loss, test_accuracy = model3.evaluate(x_test, y_test)
