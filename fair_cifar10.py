import numpy as np
import cv2
from PIL import Image, ImageOps
import tensorflow as tf
from keras.datasets import cifar10
inv_percent = 95
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


tr = True
if tr:
    data = X_train
    label = y_train
    new_data = []
    path = 'inv_plot/'

    start_ind2inv_train = int(len(data)-(len(data)*inv_percent/100))
    print("start_point2invert:", start_ind2inv_train)

    X_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
    X_train = np.array([cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in X_train])
    X_test = np.array([cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in X_test])

    for i, img in enumerate(X_train):
        if i >= start_ind2inv_train:
            X_train[i] = 255 - X_train[i]

    start_ind2inv_test = int(len(X_test) - (len(X_test) * inv_percent / 100))
    for i, img in enumerate(X_test):
        if i >= start_ind2inv_test:
            X_test[i] = 255 - X_test[i]

    X_train = (X_train / 127.5) - 1
    X_test = (X_test / 127.5) - 1
    y_train = np.array([lb[0] for lb in y_train])
    y_test = np.array([lb[0] for lb in y_test])

    np.save("x_gray_inv95_train.npy", X_train)
    np.save("y_gray_train.npy", y_train)


    np.save("x_gray_inv95_test.npy", X_test)
    np.save("y_gray_test.npy", y_test)

