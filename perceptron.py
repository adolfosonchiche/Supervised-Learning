import numpy as np

from logic_gates_enum import LogicGatesEnum


class Perceptron:
    def __init__(self, input_size, lr, epochs, gate):
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)
        self.lr = lr
        self.errors = []
        self.gate = gate

    @staticmethod
    def activation_value(x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        if self.gate is LogicGatesEnum.OR:
            x = np.insert(x, 0, 1)
            z = self.weights.T.dot(x)
            a = self.activation_value(z)
            return a
        elif self.gate is LogicGatesEnum.AND:
            z = np.dot(x, self.weights[1:]) + self.weights[0]
            a = self.activation_value(z)
            return a

    def fit(self, x, d):
        cont = 1
        for _ in range(self.epochs):
            print('iteración ' + str(cont))
            cont += 1
            self.errors = []
            for i in range(d.shape[0]):
                if self.gate is LogicGatesEnum.OR:
                    y = self.predict(x[i])
                    e = d[i] - y
                    self.weights = self.weights + self.lr * e * np.insert(x[i], 0, 1)
                    self.errors.append(e)
                elif self.gate is LogicGatesEnum.AND:
                    x = np.insert(x[i], 0, 1)
                    y = self.predict(x)
                    e = d[i] - y
                    self.weights[1:] += self.lr * e * x
                    self.weights[0] += self.lr * e
            if all(error == 0 for error in self.errors):
                print("Todos los elementos en 'errors' son cero")
                break
