import matplotlib.pyplot as plt
from ml_model.vector_model_mnist import model, X_test, X_train, Y_test, Y_train, batch_size,num_epochs

res = []

for i in range(100):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, epochs=num_epochs,
                        verbose=1)

    scores = model.evaluate(X_test, Y_test, verbose=1)

    print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))
    plt.plot(history.history['acc'])
    plt.show()
    res.append(scores[1] * 100)

print("best result")
print(max(res))
print(res)