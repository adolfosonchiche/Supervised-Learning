from logic_gates_enum import LogicGatesEnum
from perceptron import Perceptron
import numpy as np


if __name__ == '__main__':

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    ors = np.array([0, 1, 1, 1])
    perceptron_or = Perceptron(2, 0.2, 100, LogicGatesEnum.OR)
    perceptron_or.fit(X, ors)

    print("Resultado del perceptrón con la compuerta OR:")
    for x in X:
        print(x, "-->", perceptron_or.predict(x))

    #AND
    y = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    ands = np.array([0, 0, 0, 1])
    perceptron_and = Perceptron(2, 0.2, 100, LogicGatesEnum.AND)
    perceptron_and.fit(y, ands)

    print("Resultado del perceptrón con la compuerta AND:")
    for x in X:
        print(x, "-->", perceptron_and.predict(x))


