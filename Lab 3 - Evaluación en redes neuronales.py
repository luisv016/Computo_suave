"""
Red Neuronal Multicapa: FeedForward + Backpropagation + Entrenamiento
Basado en la notacion de la presentacion:

    feedforward = F( F( X . W~1 ) . W~2 )
    delta^2 = -(T - Yhat) * F'(a^2)
    delta^1 = ( delta^2 . (W^2)^T ) * F'(a^1)
    grad_W^2 = F(a^1)^T . delta^2
    grad_W^1 = X~^T . delta^1
    W~ = W~ - alpha * grad_W          (regla SGD: theta = theta - alpha * grad_theta)
"""

import numpy as np


# ------------------------------------------------------------------
# 1) Funciones de activacion y sus derivadas
# ------------------------------------------------------------------
def ReLu(x):
    return np.maximum(0, x)


def ReLu_deriv(N):
    return (N > 0).astype(float)


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoide_deriv(N):
    s = sigmoide(N)
    return s * (1.0 - s)


def lineal(x):
    return x


def lineal_deriv(N):
    return np.ones_like(N)


# Diccionario para saber, dada la funcion de activacion F, cual es F'
DERIVADAS = {
    ReLu: ReLu_deriv,
    sigmoide: sigmoide_deriv,
    lineal: lineal_deriv,
}


# ------------------------------------------------------------------
# 2) Funcion de perdida (MSE, como en la presentacion) y su derivada
# ------------------------------------------------------------------
def mse(T, Yhat):
    """1/2 (T - Yhat)^2, promediado sobre todo el batch."""
    return np.mean(0.5 * (T - Yhat) ** 2)


def mse_deriv(T, Yhat):
    """dE/dYhat = -(T - Yhat)  (la parte que multiplica a F'(a^2) en delta^2)"""
    return -(T - Yhat)


# ------------------------------------------------------------------
# 3) Capa: agrega backward() y actualizar() a lo que ya tenias
# ------------------------------------------------------------------
class Capa:
    def __init__(self, entradas, neuronas, activacion, bias=False, pesos=None):
        self.neuronas = neuronas
        self.activacion = activacion
        self.activacion_deriv = DERIVADAS[activacion]
        self.bias = bias

        if pesos is not None:
            # si ya te dan la matriz W (como en el ejercicio), se usa esa
            self.pesos = np.array(pesos, dtype=float)
        elif bias:
            # +1 fila para el peso del bias (pesos "aumentados" W~)
            self.pesos = np.random.rand(entradas + 1, neuronas) - 0.5
        else:
            self.pesos = np.random.rand(entradas, neuronas) - 0.5

    def _aumentar(self, X):
        """Agrega una columna de 1s a X si la capa usa bias -> produce X~"""
        if self.bias:
            unos = np.ones((X.shape[0], 1))
            return np.hstack([X, unos])
        return X

    def feed_forward(self, X):
        self.entrada = X                    # a^(c-1) sin aumentar (para inspeccion)
        X_aum = self._aumentar(X)           # X~
        self.entrada_aum = X_aum            # se guarda para el gradiente en backward
        N = X_aum @ self.pesos              # N = X~ . W~
        self.N = N                          # se guarda para F'(N) en backward
        self.salida = self.activacion(N)    # a^c = F(N)
        return self.salida

    def backward(self, T=None, delta_siguiente=None, W_siguiente=None, es_salida=False):
        """
        Calcula delta y el gradiente de esta capa.

        - Si es la capa de salida: delta = dE/dYhat * F'(N)
        - Si no: delta = (delta_siguiente . W_siguiente^T) * F'(N)
          (W_siguiente ya debe venir SIN la fila del bias, ver clase Red)
        """
        if es_salida:
            self.delta = mse_deriv(T, self.salida) * self.activacion_deriv(self.N)
        else:
            self.delta = (delta_siguiente @ W_siguiente.T) * self.activacion_deriv(self.N)

        # grad_W = (a^(c-1))~^T . delta   -> misma forma que self.pesos
        self.grad = self.entrada_aum.T @ self.delta
        return self.delta

    def actualizar(self, alpha, m=1):
        """SGD: W~ = W~ - alpha * grad_W  (grad promediado entre las m muestras)"""
        self.pesos = self.pesos - alpha * (self.grad / m)


# ------------------------------------------------------------------
# 4) Red: encadena varias Capas -> feedforward, backpropagation, train
# ------------------------------------------------------------------
class Red:
    def __init__(self, capas):
        self.capas = capas  # lista de objetos Capa, en orden (capa 1, capa 2, ...)

    def feedforward(self, X):
        """
        pseudo-codigo de la presentacion:
            a = X
            por cada capa C hacer: a = F(a . W~c)
            return a
        """
        a = X
        for capa in self.capas:
            a = capa.feed_forward(a)
        return a

    def backpropagation(self, T, alpha):
        """
        pseudo-codigo de la presentacion:
            si la capa es la ultima: delta = de/dyhat * F'(capa)
            si no: delta = delta_siguiente . (W_siguiente)^T
            grad_W = entrada_aumentada^T . delta
            W~ = optimizador(W~, grad_W, alpha)
        """
        n = len(self.capas)
        m = T.shape[0]  # numero de muestras, para promediar el gradiente

        # recorre las capas de la ULTIMA a la PRIMERA (por eso "back"propagation)
        for i in reversed(range(n)):
            capa = self.capas[i]
            if i == n - 1:
                capa.backward(T=T, es_salida=True)
            else:
                capa_sig = self.capas[i + 1]
                W_sig = capa_sig.pesos
                if capa_sig.bias:
                    # se descarta la fila del bias: el bias no se propaga
                    # hacia atras como conexion de neuronas
                    W_sig = W_sig[:-1, :]
                capa.backward(delta_siguiente=capa_sig.delta, W_siguiente=W_sig)

        # una vez calculados TODOS los deltas/gradientes, se actualizan los pesos
        for capa in self.capas:
            capa.actualizar(alpha, m=m)

    def train(self, X, T, epocas, tol, alpha, verbose=True, cada=100):
        """
        pseudo-codigo de la presentacion (Function Train):
            inicializar pesos (ya se hizo en el constructor de cada Capa)
            para epoca = 1 hasta epocas:
                YH = feedforward(X)
                loss = FuncionCosto(T, YH)
                si loss <= tol: salir
                backpropagation(alpha)
        """
        historial = []
        for epoca in range(1, epocas + 1):
            YH = self.feedforward(X)
            loss = mse(T, YH)
            historial.append(loss)

            if loss <= tol:
                print(f"[epoca {epoca}] tolerancia alcanzada, loss={loss:.6f}")
                break

            self.backpropagation(T, alpha)

            if verbose and (epoca % cada == 0 or epoca == 1):
                print(f"[epoca {epoca}] loss = {loss:.6f}")

        return historial

    def predecir(self, X):
        return self.feedforward(X)


# ------------------------------------------------------------------
# 5) Ejemplo 1: tu feedforward original (para verificar que sigue igual)
# ------------------------------------------------------------------
if __name__ == "__main__":
   # X = np.array([[1, 2, 3],
    #              [4, 5, 6]], dtype=float)

    X=np.ones((3, 2))*5

    W = np.array([[0,    0,   0.1, 0.2],
                  [0.3, -1,   0.4, 0.5],
                  [1,   0.6,  0.7, 0.8]])

    capa1 = Capa(entradas=3, neuronas=4, activacion=ReLu, bias=True, pesos=W)
    resultado = capa1.feed_forward(X)

    print("=== Ejemplo 1: feedforward de una sola capa (igual que antes) ===")
    print("X:\n", X)
    print("\nPesos (W):\n", capa1.pesos)
    print("\nN = X @ W:\n", capa1.N)
    print("\nY = ReLu(N):\n", resultado)

    # ----------------------------------------------------------------
    # 6) Ejemplo 2: entrenamiento completo (XOR) con 2 capas, como en
    #    el ejercicio de la presentacion: 2 capas, funcion de perdida MSE
    # ----------------------------------------------------------------
    print("\n\n=== Ejemplo 2: entrenamiento completo (XOR, 2 capas) ===")

    np.random.seed(0)

    X_xor = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]], dtype=float)

    T_xor = np.array([[0],
                       [1],
                       [1],
                       [0]], dtype=float)

    # capa 1: 2 entradas -> 4 neuronas ocultas, sigmoide, con bias
    capa_oculta = Capa(entradas=2, neuronas=4, activacion=sigmoide, bias=True)
    # capa 2: 4 entradas -> 1 salida, sigmoide, con bias
    capa_salida = Capa(entradas=4, neuronas=1, activacion=sigmoide, bias=True)

    red = Red([capa_oculta, capa_salida])

    historial = red.train(X_xor, T_xor, epocas=50000, tol=1e-4, alpha=1.0, cada=500)

    print("\nPredicciones finales:")
    print(red.predecir(X_xor).round(3))
    print("\nEsperado:")
    print(T_xor)