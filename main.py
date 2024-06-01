import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test)= mnist.load_data()
# x_train.shape= (60000, 28, 28), x_test.shape= (10000, 28, 28)
x_train=x_train/255.0
x_test=x_test/255.0

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=128, activation='relu',
                               return_sequences=True, input_shape=(28,28)))
# return_sequences=True as next layer is also lstm, if dense layer then no need
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

opt=tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
history=model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

y_pred=np.argmax(model.predict(x_test),axis=-1)
print(y_pred[0], y_test[0])
print(y_pred[10], y_test[10])
print(y_pred[-5], y_test[-5])

cm=confusion_matrix(y_test, y_pred)
print(cm)
acc_cm= accuracy_score(y_test, y_pred)
print(acc_cm)

epoch_range=range(1,11)
plt.plot(epoch_range, history.history['sparse_categorical_accuracy'])
plt.plot(epoch_range, history.history['val_sparse_categorical_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epochs")
plt.legend(['Train', 'val'], loc='upper left')
plt.show()

plt.plot(epoch_range, history.history['loss'])
plt.plot(epoch_range, history.history['val_loss'])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(['Train', 'val'], loc='upper left')
plt.show()