import numpy as np

class Capa:
    def __init__(self, entradas, neuronas, activacion, bias=False, pesos=None):
        self.neuronas = neuronas
        self.activacion = activacion
        self.bias = bias

        if pesos is not None:
            # si ya te dan la matriz W (como en el ejercicio), se usa esa
            self.pesos = np.array(pesos, dtype=float)
        elif bias:
            self.pesos = np.random.rand(entradas + 1, neuronas)
        else:
            self.pesos = np.random.rand(entradas, neuronas)

    def feed_forward(self, X):
        self.entrada = X
        N = X @ self.pesos      # <- aqui pasa la multiplicacion
        self.N = N               # la guardamos para poder verla despues
        Y = self.activacion(N)
        return Y


def ReLu(x):
    return np.maximum(0, x)


X = np.array([[1, 2, 3],
              [4, 5, 6]])

W = np.array([[0,    0,   0.1, 0.2],
              [0.3, -1,   0.4, 0.5],
              [1,   0.6,  0.7, 0.8]])

capa1 = Capa(entradas=3, neuronas=4, activacion=ReLu, bias=False, pesos=W)
resultado = capa1.feed_forward(X)

print("X:")
print(X)

print("\nPesos (W):")
print(capa1.pesos)

print("\nMultiplicacion N = X @ W:")
print(capa1.N)

print("\nResultado despues de ReLu (Y):")
print(resultado)