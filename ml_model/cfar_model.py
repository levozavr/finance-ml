import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from PreProcessors.csv_pre_processor_assim_heatmap_relation import PreProcessor
import matplotlib.pyplot as plt
import os
import numpy as np

batch_size = 200
num_classes = 3
epochs = 20
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

PP = PreProcessor(filename="../index/FX_USDKRW.csv")
PP.start(grade=1.005)
"""
train_x: обучающие данные содержащие в себе heatmap 20X20
train_y: обучающие данные(предсказания) содержащие в себе 7-дневные тренды"""

x_train, y_train = PP.get_train()
"""
test_x: тренировачные данные содержащие в себе heatmap 20X20
test_y: тренировачные данные(предсказания) содержащие в себе 7-дневные тренды"""

x_test, y_test = PP.get_test()

x_val, y_val = PP.get_val()


def distribution(y, st=""):
    a0,b0,c0 =0,0,0
    for a,b,c in y:
        if a ==1:
            a0+=1
        if b == 1:
            b0+=1
        if c ==1:
            c0+=1
    print(st)
    print(f"dec: {int(100*a0/len(y))}%, st: {int(100*b0/len(y))}%, inc: {int(100*c0/len(y))}%")


distribution(y_train, "train:")
distribution(y_val, "validation:")
distribution(y_test, "test:")

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(64, (5, 5), padding='same',
                 input_shape=(25, 25, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.75))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.75))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('softmax'))
model.add(Dense(64))
model.add(Activation('softmax'))
model.add(Dropout(0.75))
model.add(Dense(3, activation='softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

# Let's train the model using RMSprop
if __name__ == "__main__":
    model.compile(loss='mean_squared_error',
                  optimizer="adam",
                  metrics=['accuracy'])

    """
    history: данные полученные во время обучения сети, необходимые для построения различных графиков"""
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=[x_val, y_val])
    """if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


    Данные полученные после тестирования сети - точность работы на тренировчном множестве """
    res = model.predict(x_test, 200)
    a0,a1,a2 = 0,0,0
    for val in res:
        if max(val) == val[0]:
            a0+=1
        if max(val) == val[1]:
            a2+=1
        if max(val) == val[2]:
            a2+=1
    print(f"dec: {a0}, st: {a1}, inc: {a2}")
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))
    plt.plot(np.array(history.history['acc']))
    plt.plot(np.array(history.history['val_acc']))
    plt.show()
