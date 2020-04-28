import numpy as np


# object is a base for all classes.
# It has the methods that are common to all instances of Python classes.
# No need to pay attention to this for now.
# More about base class object: https://blog.csdn.net/qq_36556893/article/details/90770433
class Activation(object):
    def __init__(self, activate):
        self.forward_func = Activation._forward_by_name(activate)
        self.backward_func = Activation._backward_by_name(activate)

    @classmethod
    def _sigmoid_forward(cls, X):
        return 1 / (1 + np.exp(-X))

    @classmethod
    def _sigmoid_backward(cls, layer_output):
        return layer_output * (1 - layer_output)

    @classmethod
    def _none_forward(cls, X):
        return X

    @classmethod
    def _none_backward(cls, layer_output):
        return 1

    @classmethod
    def _forward_by_name(cls, activate):
        if activate is None:
            return cls._none_forward
        elif activate.lower() == 'sigmoid':
            return cls._sigmoid_forward

    @classmethod
    def _backward_by_name(cls, activate):
        if activate is None:
            return cls._none_backward
        elif activate.lower() == 'sigmoid':
            return cls._sigmoid_backward

    def forward_activation(self, layer_input):
        return self.forward_func(layer_input)

    def backward_gradient(self, layer_output):
        return self.backward_func(layer_output)

    # use __str__ function to output the information of the class when using print() function,
    # or any other case when we want a str instance of the class
    def __str__(self):
        if self.forward_func == Activation._none_forward:
            s = '<None Activation>'
        elif self.forward_func == Activation._sigmoid_forward:
            s = '<Sigmoid Activation>'
        else:
            raise NotImplementedError('Not supported activation function: {}'.format(self.forward_func.__name__))
        return s

    __repr__ = __str__


class FCLayer(object):
    def __init__(self, units, use_bias=True, activation=None):
        self.units = units
        self.use_bias = use_bias
        self.bias = np.zeros((self.units, 1)) if self.use_bias else None
        self.activator = Activation(activation)
        self.input, self.output, self.delta = None, None, None
        self.w_update, self.b_update = None, None

    def _add_weights(self, pre_units=None):
        self.weights = None if pre_units is None else np.random.random((self.units, pre_units))

    def forward(self, input_x):
        self.input = np.reshape(input_x, (-1, 1))
        self.output = self.activator.forward_activation(np.dot(self.weights, self.input) + self.bias)
        return self.output

    def backward(self, delta_y):
        self.gradient = self.activator.backward_gradient(self.input)
        self.delta = self.gradient * np.dot(self.weights.T, delta_y)
        self.w_update = np.dot(delta_y, self.input.T)
        self.b_update = delta_y
        return self.delta

    def update(self, learning_rate):
        self.weights += learning_rate * self.w_update
        self.bias += learning_rate * self.b_update


class SequentialModel(object):
    def __init__(self):
        self.layers = []

    def add_fc_layer(self, units, use_bias=False, activation=None):
        layer = FCLayer(units, use_bias, activation)
        layer._add_weights(self.layers[-1].units if self.layers else None)
        self.layers.append(layer)

    def fit(self, train_data, train_labels, epoch, learning_rate=0.01):
        train_data, train_labels = np.asarray(train_data), np.asarray(train_labels)
        for i in range(epoch):
            for x, y in zip(train_data, train_labels):
                self._train_on_one_sample(x, y, learning_rate)
            print('Evaluating Model after epoch {}...'.format(i + 1))
            acc, loss = self.evaluate(train_data, train_labels)
            print('training acc: {}, training loss: {}'.format(acc, loss))
        print('Training Finished. The prediction of training set is:\n {}'.format(self.predict_labels(train_data)))
        print('The groundtruths of training set is:\n {}'.format(np.argmax(train_labels, axis=1)))

    def _train_on_one_sample(self, x, y, learning_rate):
        self._predict_on_one_sample(x)
        self._gradient_dencent(y)
        self._update(learning_rate)

    def predict_labels(self, X):
        return np.argmax(self.predict(X), axis=1).squeeze()

    def predict(self, X):
        if np.ndim(X) == 1:
            X = np.reshape(X, (-1, 1))
        return np.asarray([self._predict_on_one_sample(x) for x in X])

    def _predict_on_one_sample(self, x):
        assert np.size(x) == self.layers[0].units, 'input size :{} != units: {}'.format(np.size(x),
                                                                                        self.layers[0].units)
        for layer in self.layers[1:]:
            x = layer.forward(x)
        return x

    def evaluate(self, data, labels):
        acc = self.get_acc(self.predict_labels(data), labels)
        loss = self.get_loss(self.predict(data), labels)
        return acc, loss

    def get_acc(self, predict_labels, groundtruths):
        return np.sum(np.argmax(groundtruths, axis=1) == predict_labels) / len(predict_labels)

    def get_loss(self, predicts, groundtruths):
        return np.sum([self._get_loss_on_one_sample(p, g) for p, g in zip(predicts, groundtruths)])

    def _get_loss_on_one_sample(self, predict, groundtruth):
        return np.sum((groundtruth.reshape(-1, 1) - predict) ** 2) / 2

    def _gradient_dencent(self, label):
        delta = self._get_output_delta_on_one_sample(self.layers[-1].output, label)
        for layer in self.layers[:0:-1]:
            delta = layer.backward(delta)

    def _get_output_delta_on_one_sample(self, output, groundtruth):
        return self.layers[-1].activator.backward_gradient(output) * (groundtruth.reshape(-1, 1) - output)

    def _update(self, learning_rate):
        for layer in self.layers[1:]:
            layer.update(learning_rate)


if __name__ == '__main__':
    model = SequentialModel()
    model.add_fc_layer(2)
    model.add_fc_layer(10, activation='sigmoid', use_bias=True)
    model.add_fc_layer(3, activation='sigmoid', use_bias=True)

    X = [[0, 3], [5, 6], [-1, 11], [-1, 3]]
    y = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]

    model.fit(X, y, 1000, 0.1)
