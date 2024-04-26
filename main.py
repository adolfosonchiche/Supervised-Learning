from logic_gates_enum import LogicGatesEnum
from perceptron import Perceptron
import numpy as np


if __name__ == '__main__':
    #entrada de datos
    data_in = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # perceptron para una compuerta OR
    ors = np.array([0, 1, 1, 1])
    perceptron_or = Perceptron(2, 0.3, 100, LogicGatesEnum.OR)
    perceptron_or.fit(data_in, ors)

    print("Resultado del perceptrón con la compuerta OR:")
    for x in data_in:
        print(x, "-->", perceptron_or.predict(x))

    #perceptron para una compuerta AND
    ands = np.array([0, 0, 0, 1])
    perceptron_and = Perceptron(2, 0.2, 100, LogicGatesEnum.AND)
    perceptron_and.fit(data_in, ands)

    print("Resultado del perceptrón con la compuerta AND:")
    for x in data_in:
        print(x, "-->", perceptron_and.predict(x))


