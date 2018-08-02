import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("---- System ----")
print(tf.__version__)
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))

print("---- Setup Data ----")
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0
test_images = test_images / 255.0

print("---- Setup Layers ----")
EPOCHS = 1
NODES = 32

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(NODES, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print("---- Training ----")
startTime = time.time()
model.fit(train_images, train_labels, epochs=EPOCHS)
endTime = time.time()

print("---- Testing ----")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: {}'.format(test_acc))
predictions = model.predict(test_images)

print("---- Result ----")

missed = 0
success = 0
for i in range(predictions.shape[0]):
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label != true_label:
        missed += 1
    else:
        success += 1

print("Missed: {}".format(missed))
print("Success: {}".format(success))
print("Accuracy: {:.3} %".format((1.0 - float(missed) / float(success)) * 100.0))
print("Epochs: {}".format(EPOCHS))
print("Nodes (middle layer): {}".format(NODES))
print("Time: {:.2} seconds".format(endTime - startTime))




# ---------------------------------------------------- Sci Views



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

# SciView test object
for i in range(25):
    displayObject = i
    img = test_images[displayObject]
    plt.figure()
    plt.subplot(121)
    plt.title("{} [{}]".format(class_names[test_labels[displayObject]], displayObject))
    plt.imshow(img)

    # SciView test result
    img = (np.expand_dims(img, 0))
    predictions = model.predict(img)[0]
    plt.subplot(122)
    plt.title("{} [{}]".format(class_names[test_labels[displayObject]], displayObject))
    for i in range(predictions.size):
        plt.bar(i, predictions[i], 1)
    plt.xlabel('Label')
    plt.ylabel('Probability')
    plt.locator_params(nbins=11, axis='x')
    plt.axis((0, 9, 0, 1))
    plt.xticks(range(10), class_names, rotation=60)
    plt.margins(10)
    plt.show()