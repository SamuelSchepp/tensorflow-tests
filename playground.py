# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# SciView Learning Grid
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# SciView Colorbar
plt.figure()
plt.imshow(train_images[0])
plt.title(class_names[train_labels[0]])
plt.colorbar()
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

# SciView Tests
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
      plt.xlabel("{}".format(class_names[predicted_label]), color='green')
    else:
      plt.xlabel("{} (is: {})".format(class_names[predicted_label], class_names[true_label]), color='red')
plt.show()

img = test_images[0]

# SciView test object
plt.figure()
plt.title(class_names[test_labels[0]])
plt.imshow(img)
plt.colorbar()
plt.show()

img = (np.expand_dims(img, 0))
predictions = model.predict(img)[0]
print(predictions)

# SciView test result
plt.figure()
plt.title(class_names[test_labels[0]])
for i in range(predictions.size):
    plt.bar(i, predictions[i], 1)
plt.xlabel('Label')
plt.ylabel('Probability')
plt.locator_params(nbins=11, axis='x')
plt.axis((0, 9, 0, 1))
plt.xticks(range(10), class_names, rotation=45)
plt.margins(10)
plt.show()