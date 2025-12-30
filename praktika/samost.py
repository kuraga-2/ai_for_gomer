import numpy as np
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

def main():
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train[:200].reshape((200, 28*28)).astype('float32') / 255
    y_train = to_categorical(y_train[:200])

    model = models.Sequential([
        layers.Dense(64, kernel_initializer='glorot_uniform', input_shape=(784,), activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy')#metrics=['accuracy']
    w_before, b_before = model.layers[0].get_weights()
    model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1)
    w_after, b_after = model.layers[0].get_weights()

    print("Минимальное значение весов:", np.min(w_after))
    print("Максимальное значение весов:", np.max(w_after))
    print("Среднее значение весов:", np.mean(w_after))
    print("Дисперсия весов:", np.var(w_after))


if __name__ == "__main__":
    main()

