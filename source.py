import os
import os.path
import pickle
import gzip
import os
import numpy as np

from PIL import Image

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

mnist_files = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]


class Mnist(object):
    def __init__(self, file_list=mnist_files):
        self.name = 'MNIST_dataset'
        self.data = dict()
        self.data['train_img'] = self.load_img(file_list[0])
        self.data['train_label'] = self.load_label(file_list[1])
        self.data['test_img'] = self.load_img(file_list[2])
        self.data['test_label'] = self.load_label(file_list[3])

    def load_img(self, file_name):
        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_name, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, img_size)
        print("Done")
        return data

    def load_label(self, file_name):
        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_name, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        print("Done")
        return labels

    def img_show(self, data_name, num):
        img = self.data[data_name][num]
        pil_img = Image.fromarray(np.uint8(img.reshape(28, 28)))
        pil_img.show()


class ThreeLayerNet(object):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_init_std=0.01):
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size2, output_size)
        self.params['b3'] = np.zeros(output_size)

    def predict(self, network, x):
        w1, w2, w3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(x, w1) + b1
        z1 = Function.sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = Function.sigmoid(a2)
        a3 = np.dot(z2, w3) + b3
        y = Function.softmax(a3)
        return y

    def lose(self, input_data, train_data):
        y = self.predict(input_data)
        batch_size = y.shape(1, train_data.size)
        return -np.sum(np.log(y[np.average(batch_size), train_data])) / batch_size


class Function(object):
    def __init__(self):
        self.name = ""

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))


if __name__ == '__main__':
    mnist = Mnist()

    mnist.img_show("train_img", 1)

    print(mnist.data['train_label'][0])
