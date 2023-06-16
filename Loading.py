from pathlib import Path
import numpy as np
import cv2

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def get_train_data():
    train_data = Path.cwd() / 'images' / 'train'

    train_X = []
    train_y = []

    # Append image pixel array and label to lists
    for n in numbers:
        for i in (train_data / n).iterdir():
            train_X.append(cv2.imread(str(i), cv2.IMREAD_UNCHANGED))
            train_y.append(int(n))

    train_X = np.array(train_X)
    train_X = train_X.astype(np.float32) / 255
    return train_X, np.array(train_y)

def get_test_data():
    test_data = Path.cwd() / 'images' / 'test'

    test_X = []
    test_y = []

    # Append image pixel array and label to lists
    for n in numbers:
        for i in (test_data / n).iterdir():
            test_X.append(cv2.imread(str(i), cv2.IMREAD_UNCHANGED))
            test_y.append(int(n))

    test_X = np.array(test_X)
    test_X = test_X.astype(np.float32) / 255
    return test_X, np.array(test_y)