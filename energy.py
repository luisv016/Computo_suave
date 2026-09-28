import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1) ACTIVACIONES
# ============================================================
def ReLu(x):
    return np.maximum(0, x)

def ReLu_deriv(x):
    return (x > 0).astype(float)

def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoide_deriv(x):
    s = sigmoide(x)
    return s * (1.0 - s)

# NUEVA: identidad (para regresión en la capa de salida)
def identidad(x):
    return x

def identidad_deriv(x):
    return np.ones_like(x)

DERIVADAS = {
    ReLu: ReLu_deriv,
    sigmoide: sigmoide_deriv,
    identidad: identidad_deriv,
}


# ============================================================
# 2) PÉRDIDA
# ============================================================
def mse(T, Yhat):
    return np.mean(0.5 * (T - Yhat) ** 2)

def mse_deriv(T, Yhat):
    return -(T - Yhat)


# ============================================================
# 3) CAPA
# ============================================================
class Capa:
    def __init__(self, entradas, neuronas, activacion, bias=False, pesos=None):
        self.activacion = activacion
        self.activacion_deriv = DERIVADAS[activacion]
        self.bias = bias
        if pesos is not None:
            self.pesos = np.array(pesos, dtype=float)
        else:
            filas = entradas + 1 if bias else entradas
            # Inicialización tipo Xavier (mejor para redes profundas)
            limite = np.sqrt(6.0 / (entradas + neuronas))
            self.pesos = np.random.uniform(-limite, limite, (filas, neuronas))

    def _aumentar(self, X):
        if self.bias:
            return np.hstack([X, np.ones((X.shape[0], 1))])
        return X

    def feed_forward(self, X):
        self.entrada_aum = self._aumentar(X)
        self.N = self.entrada_aum @ self.pesos
        self.salida = self.activacion(self.N)
        return self.salida

    def backward(self, T=None, delta_siguiente=None, W_siguiente=None, es_salida=False):
        if es_salida:
            self.delta = mse_deriv(T, self.salida) * self.activacion_deriv(self.N)
        else:
            self.delta = (delta_siguiente @ W_siguiente.T) * self.activacion_deriv(self.N)
        self.grad = self.entrada_aum.T @ self.delta
        return self.delta

    def actualizar(self, alpha, m=1):
        self.pesos -= alpha * (self.grad / m)


# ============================================================
# 4) RED
# ============================================================
class Red:
    def __init__(self, capas):
        self.capas = capas

    def feedforward(self, X):
        a = X
        for capa in self.capas:
            a = capa.feed_forward(a)
        return a

    def backpropagation(self, T, alpha):
        n = len(self.capas)
        m = T.shape[0]
        for i in reversed(range(n)):
            capa = self.capas[i]
            if i == n - 1:
                capa.backward(T=T, es_salida=True)
            else:
                capa_sig = self.capas[i + 1]
                W_sig = capa_sig.pesos
                if capa_sig.bias:
                    W_sig = W_sig[:-1, :]
                capa.backward(delta_siguiente=capa_sig.delta, W_siguiente=W_sig)

        for capa in self.capas:
            capa.actualizar(alpha, m=m)

    def train(self, X, T, epocas, tol, alpha, cada=500):
        for epoca in range(1, epocas + 1):
            YH = self.feedforward(X)
            loss = mse(T, YH)

            if epoca % cada == 0:
                print(f"Epoca {epoca} | Pérdida (MSE): {loss:.6f}")

            if loss <= tol:
                print(f"Epoca {epoca} | Pérdida (MSE): {loss:.6f} (tolerancia alcanzada)")
                return

            self.backpropagation(T, alpha)

        print(f"Epoca {epocas} | Pérdida (MSE) Final: {loss:.6f}")

    def predecir(self, X):
        return self.feedforward(X)


# ============================================================
# 5) CARGA DE DATOS Y ENTRENAMIENTO
# ============================================================
if __name__ == "__main__":
    np.random.seed(0)

    # -------- Cargar CSV --------
    df = pd.read_csv("ENB2012_data.csv")
    print("Shape dataset:", df.shape)
    print("Columnas:", df.columns.tolist())

    X_raw = df[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].values.astype(float)
    Y_raw = df[["Y1", "Y2"]].values.astype(float)

    # -------- Escalar --------
    scaler_X = StandardScaler().fit(X_raw)
    scaler_Y = StandardScaler().fit(Y_raw)

    X = scaler_X.transform(X_raw)
    Y = scaler_Y.transform(Y_raw)

    # -------- Split train/test --------
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.4, random_state=42
    )

    # -------- Definir la red --------
    # 8 entradas -> 16 ocultas (sigmoide) -> 8 ocultas (sigmoide) -> 2 salidas (identidad)
    red = Red([
        Capa(entradas=8,  neuronas=16, activacion=sigmoide,  bias=True),
        Capa(entradas=16, neuronas=8,  activacion=sigmoide,  bias=True),
        Capa(entradas=8,  neuronas=2,  activacion=identidad, bias=True),
    ])

    # -------- Entrenar --------
    red.train(X_train, Y_train, epocas=5000, tol=1e-6, alpha=0.1, cada=500)

    # -------- Evaluar --------
    Yhat_train = red.predecir(X_train)
    Yhat_test  = red.predecir(X_test)

    # Volver a escala real
    Yhat_train_real = scaler_Y.inverse_transform(Yhat_train)
    Yhat_test_real  = scaler_Y.inverse_transform(Yhat_test)
    Y_train_real    = scaler_Y.inverse_transform(Y_train)
    Y_test_real     = scaler_Y.inverse_transform(Y_test)

    print("\n================= MÉTRICAS =================")
    for nombre, yt, yp in [
        ("Train Y1 (heating)", Y_train_real[:, 0], Yhat_train_real[:, 0]),
        ("Train Y2 (cooling)", Y_train_real[:, 1], Yhat_train_real[:, 1]),
        ("Test  Y1 (heating)", Y_test_real[:, 0],  Yhat_test_real[:, 0]),
        ("Test  Y2 (cooling)", Y_test_real[:, 1],  Yhat_test_real[:, 1]),
    ]:
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2   = r2_score(yt, yp)
        print(f"{nombre}: RMSE = {rmse:.3f} | R² = {r2:.4f}")

    # -------- Predicción de una muestra nueva --------
    nueva = np.array([[0.98, 514.5, 294.0, 110.25, 7.0, 2, 0.0, 0]])
    nueva_esc = scaler_X.transform(nueva)
    pred_esc  = red.predecir(nueva_esc)
    pred_real = scaler_Y.inverse_transform(pred_esc)
    print("\nPredicción nueva muestra (Y1, Y2):", pred_real.round(2))
    print("Valor real esperado (fila 1):     [15.55, 21.33]")

    # -------- Importancia de las variables --------
    W1 = red.capas[0].pesos  # (9, 16): 8 features + bias, 16 neuronas
    importancias = np.abs(W1[:-1, :]).mean(axis=1)
    print("\n================= IMPORTANCIA DE X =================")
    for i, imp in enumerate(importancias, start=1):
        print(f"X{i}: {imp:.3f}")