from random import random

import numpy as np

from logic_gates_enum import LogicGatesEnum


class Perceptron:
    def __init__(self, input_size, umbral, epochs, gate):
        self.epochs = epochs
        self.weights = np.zeros(input_size + 1)
        self.weights = data_in = np.array([
            random() for _ in range(input_size+1)
        ])
        self.umbral = umbral
        self.errors = []
        self.gate = gate

    @staticmethod
    def activation_value(x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.weights.T.dot(x)   #sumar sesgo
        a = self.activation_value(z)
        return a

    def fit(self, x, d):
        cont = 1
        for _ in range(self.epochs):
            print('iteración ' + str(cont))
            cont += 1
            self.errors = []
            for i in range(d.shape[0]):
                result = self.predict(x[i])
                e = d[i] - result
                self.weights = self.weights + self.umbral * e * np.insert(x[i], 0, 1)
                self.errors.append(e)
            if all(error == 0 for error in self.errors):
                print("Todos los elementos en 'errors' son cero")
                break
