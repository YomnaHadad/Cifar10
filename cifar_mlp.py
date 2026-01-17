import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import cifar10
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Input


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model = Sequential([
    # The input shape should be the flattened image size (32*32*3)

    keras.Input(shape=(x_train.shape[1] * x_train.shape[2] * x_train.shape[3],)),
    keras.layers.Dense(1024, activation='relu'),
    BatchNormalization(),

    keras.layers.Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    keras.layers.Dense(256, activation='relu'),
    BatchNormalization(),
    keras.layers.Dense(128, activation='relu'),
    Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])

if __name__ == "__main__" :
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  print(f"x_train shape: {x_train.shape}")
  print(f"y_train shape: {y_train.shape}")
  print(f"x_test shape: {x_test.shape}")
  print(f"y_test shape: {y_test.shape}")

  x_train = x_train.reshape(x_train.shape[0], -1)
  x_test  = x_test.reshape(x_test.shape[0], -1)

  x_train = x_train / 255.
  x_test = x_test / 255.

  callbacks = [   # boosting the accuracy
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
  ]

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
  print(model.summary())

  num_epochs = 50
  model.fit(x_train, y_train, epochs= num_epochs, batch_size=32, verbose=2, validation_split=0.1, callbacks=callbacks)

  ### Testing
  test_loss, test_accuracy = model.evaluate(x_test, y_test)
