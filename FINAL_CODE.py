import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()

to_categorical(3, num_classes=10)

temp = []
for i in range(len(y_train)):
  temp.append(to_categorical(y_train[i], num_classes=10))
y_train = np.array(temp)
temp = []
for i in range(len(y_test)):    
  temp.append(to_categorical(y_test[i], num_classes=10))
y_test = np.array(temp)

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid')) ###TO CHANGE THE ACTIVATION FUNCTION, JUST CHANGE 'sigmoid' per 'softmax' here

model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc'])

history = model.fit(X_train, y_train, epochs=200, 
          validation_data=(X_test,y_test), verbose = 2)

epoc= list(range(1,201))

results = model.evaluate(X_test, y_test, batch_size=128)

def plot_acc():
    xpoints = np.array(epoc)
    ypoints = np.array(history.history['val_acc'])
    plt.xlabel('Época', fontsize=15)
    plt.ylabel('Acurácia', fontsize=15)
    plt.plot(xpoints, ypoints)
    plt.show()

def plot_loss():
    xpoints = np.array(epoc)
    ypoints = np.array(history.history['val_loss'])
    plt.xlabel('Época', fontsize=15)
    plt.ylabel('Entropia Cruzada', fontsize=15)
    plt.plot(xpoints, ypoints)
    plt.show()





