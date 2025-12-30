from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
import numpy as np
from keras.preprocessing import image

def model_main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((60000, 28*28)).astype("float32") / 255
    x_test = x_test.reshape((10000, 28*28)).astype("float32") / 255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    optimizers = ["SGD", "Adam", "RMSprop"]
    results = {}
    trained_models = []

    for opt in optimizers:
        print(f"\n=== Обучение с оптимизатором {opt.upper()} ===")
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(28*28,)),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer=opt,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        model.fit(
            x_train, y_train,
            epochs=10, 
            batch_size=128,
            validation_split=0.1,
            verbose=2
        )
        
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        results[opt] = test_acc
        trained_models.append(model)

    print("\nСравнение точности на тесте:")
    for opt, acc in results.items():
        print(f"{opt.upper()}: {acc:.4f}")

    return trained_models


def img(model, img_path):
    img = image.load_img(img_path, color_mode="grayscale", target_size=(28,28))
    img_array = image.img_to_array(img)
    img_array = img_array.reshape((1, 28*28)).astype('float32')/255

    prediction = model.predict(img_array)
    predicted = np.argmax(prediction)
    print("Предсказанная цифра:", predicted)
