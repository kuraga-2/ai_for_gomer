import numpy as np
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

def build_model(random_init=True):
    model = models.Sequential([
        layers.Dense(64, kernel_initializer='glorot_uniform', input_shape=(784,), activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    if not random_init:
        w = np.ones((784, 64), dtype=np.float32)
        b = np.ones((64,), dtype=np.float32)
        model.layers[0].set_weights([w, b])
    return model

def train_and_compare(random_init=True):
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train[:200].reshape((200, 28*28)).astype('float32') / 255
    y_train = to_categorical(y_train[:200])

    model = build_model(random_init=random_init)
    w_before, b_before = model.layers[0].get_weights()
    model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
    w_after, b_after = model.layers[0].get_weights()
    delta = w_after - w_before

    print("Случайная" if random_init else "Одинаковая", "инициализация")
    print("Минимальное измененение веса:", np.min(delta))
    print("Максимальное изменение веса:", np.max(delta))
    print("Среднее изменение веса:", np.mean(delta))

def main():
    train_and_compare(random_init=True)   # случайная
    train_and_compare(random_init=False)  # одинаковая

if __name__ == "__main__":
    main()
