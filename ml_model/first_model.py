from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from PreProcessors.csv_pre_processor import PreProcessor
import matplotlib.pyplot as plt

model = Sequential()
model.add(Conv2D(70, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(20, 20, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(200, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 16
PP = PreProcessor(filename="../index/avg_index.csv")
PP.start()
"""
train_x: обучающие данные содержащие в себе heatmap 20X20
train_y: обучающие данные(предсказания) содержащие в себе 7-дневные тренды
"""
train_x, train_y = PP.get_train()
"""
test_x: тренировачные данные содержащие в себе heatmap 20X20
test_y: тренировачные данные(предсказания) содержащие в себе 7-дневные тренды
"""
test_x, test_y = PP.get_test()
"""
history: данные полученные во время обучения сети, необходимые для построения различных графиков
"""
history = model.fit(x=train_x, y=train_y, batch_size=2000, epochs=20, validation_split=0.2, verbose=2)
"""
Данные полученные после тестирования сети - точность работы на тренировчном множестве
"""
scores = model.evaluate(test_x, test_y, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.show()


#Коментраии к переменным
#Работа с векторными данными
#Модель для cfar10