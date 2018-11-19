from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
from PreProcessors.csv_pre_processor_vector import PreProcessor
import matplotlib.pyplot as plt


batch_size = 200
num_epochs = 200
hidden_size = 512

PP = PreProcessor('../index/avg_index.csv')
PP.start(grade=20)
"""
train_x: обучающие данные содержащие в себе вектор 20X1
train_y: обучающие данные(предсказания) содержащие в себе 7-дневные тренды"""

X_train, Y_train = PP.get_train()
"""
test_x: тренировачные данные содержащие в себе вектор 20X1
test_y: тренировачные данные(предсказания) содержащие в себе 7-дневные тренды"""

X_test, Y_test = PP.get_test()

inp = Input(shape=(20,))
hidden_1 = Dense(hidden_size, activation='relu')(inp)
hidden_2 = Dense(hidden_size, activation='relu')(hidden_1)
out = Dense(3, activation='softmax')(hidden_2)

model = Model(input=inp, output=out)

if __name__ == "__main__":
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
              batch_size=batch_size, epochs=num_epochs,
              verbose=1)

    scores = model.evaluate(X_test, Y_test, verbose=1)


    print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))
    plt.plot(history.history['acc'])
    plt.show()

