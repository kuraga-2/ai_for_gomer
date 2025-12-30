import numpy as np
from keras import models, layers, initializers
from keras.datasets import mnist
from keras.utils import to_categorical

def main():
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train[:200].reshape((200, 28*28)).astype('float32') / 255
    y_train = to_categorical(y_train[:200])

    initializer = initializers.RandomUniform(minval=-1e-6, maxval=1e-6) 

    model = models.Sequential([
        layers.Dense(64, kernel_initializer=initializer, input_shape=(784,), activation='relu'),
        layers.Dense(10, kernel_initializer=initializer, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)
    test_example = x_train[:5]

    for i in range(3): 
        outputs = model.predict(test_example, verbose=0)
        print(f"Прогон {i+1}:")
        print(outputs)

    avg_outputs = np.mean([model.predict(test_example, verbose=0) for _ in range(5)], axis=0)
    print("Средние выходы по 5 прогонам:")
    print("Среднее значение весов:", np.mean(avg_outputs))

if __name__ == "__main__":
    main()

