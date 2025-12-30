from keras import models, layers, initializers
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def main():
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train[:200].reshape((200, 28*28)).astype('float32') / 255
    y_train = to_categorical(y_train[:200])

    initializer = initializers.GlorotUniform()

    frozen_model = models.Sequential([
        layers.Dense(64, kernel_initializer=initializer, input_shape=(784,), activation='relu', trainable=False),
        layers.Dense(10, kernel_initializer=initializer, activation='softmax')
    ])
    frozen_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    frozen_history = frozen_model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=0)

    trainable_model = models.Sequential([
        layers.Dense(64, kernel_initializer=initializer, input_shape=(784,), activation='relu'),
        layers.Dense(10, kernel_initializer=initializer, activation='softmax')
    ])
    trainable_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    trainable_history = trainable_model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=0)

    print("Точность с замороженным слоем:", frozen_history.history['accuracy'][-1])
    print("Точность полностью обучаемой модели:", trainable_history.history['accuracy'][-1])

    frozen_loss = frozen_history.history['loss']
    trainable_loss = trainable_history.history['loss']

    plt.figure(figsize=(8,5))
    plt.plot(frozen_loss, label='Замороженный первый слой')
    plt.plot(trainable_loss, label='Полностью обучаемая')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.title('Loss по эпохам')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()