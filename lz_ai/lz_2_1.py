import numpy as np
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def build_model():
    model = models.Sequential([
        layers.Dense(64, kernel_initializer='glorot_uniform', input_shape=(784,), activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((60000, 784)).astype('float32') / 255
    x_test = x_test.reshape((10000, 784)).astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = build_model()
    history1 = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=2)
    model.save_weights("mnist.weights.h5")

    model2 = build_model()
    model2.load_weights("mnist.weights.h5")
    history2 = model2.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=2)
    loss1, acc1 = model.evaluate(x_test, y_test, verbose=0)
    loss2, acc2 = model2.evaluate(x_test, y_test, verbose=0)

    print("Первое обучение: loss=%.4f, acc=%.4f" % (loss1, acc1))
    print("Второе обучение (после перезапуска): loss=%.4f, acc=%.4f" % (loss2, acc2))
    
    loss1 = history1.history['loss']
    loss2 = history2.history['loss']
    loss_diff = [l1 - l2 for l1, l2 in zip(loss1, loss2)]

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(loss_diff)+1), loss_diff, marker='o', label='Разница loss (первое - второе)')
    plt.title('Разница loss по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Разница loss')
    plt.grid(True)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()