import numpy as np


def ReLu(x):
    return np.maximum(0, x)

def ReLu_derivada(x):
    return np.where(x > 0, 1.0, 0.0)


def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def mse_loss_derivada(y_pred, y_true):
    
    return 2 * (y_pred - y_true) / y_pred.size


class Capa:
    def __init__(self, entradas, neuronas, activacion, activacion_derivada, bias=False, pesos=None):
        self.neuronas = neuronas
        self.activacion = activacion
        self.activacion_derivada = activacion_derivada
        self.bias = bias
        
        if pesos is not None:
            self.pesos = np.array(pesos, dtype=float)
        elif bias:
            self.pesos = np.random.rand(entradas + 1, neuronas)
        else:
            
            self.pesos = np.random.rand(entradas, neuronas) * 0.1
            
    def feed_forward(self, X):
        self.entrada = X
        N = X @ self.pesos 
        self.N = N 
        Y = self.activacion(N)
        return Y

    def backpropagation(self, alpha, delta_siguiente=None, W_siguiente=None, es_ultima=False, d_error_d_pred=None):
        
        if es_ultima:
            self.delta = d_error_d_pred * self.activacion_derivada(self.N)
        else:
            self.delta = (delta_siguiente @ W_siguiente.T) * self.activacion_derivada(self.N)
        
        
        grad_W = self.entrada.T @ self.delta
        
       
        self.pesos = self.pesos - alpha * grad_W
        
        return self.delta



X = np.ones((3, 2)) * 5
y = np.zeros((3, 2))


np.random.seed(42) 


capa_salida = Capa(entradas=2, neuronas=2, activacion=ReLu, activacion_derivada=ReLu_derivada, bias=False)

print("=== DATOS DE INICIALIZACIÓN ===")
print("Matriz de Entrada X (3x2):\n", X)
print("\nPesos Iniciales W (2x2):\n", capa_salida.pesos)
print("\nMatriz Objetivo y (3x2):\n", y)


y_pred = capa_salida.feed_forward(X)
loss_inicial = mse_loss(y_pred, y)

print("\n=== COMPROBACIÓN: PASADA HACIA ADELANTE ===")
print("Multiplicación neta N (X @ W):\n", capa_salida.N)
print("\nPredicción final Y (tras ReLU):\n", y_pred)
print(f"\nError (Loss) inicial: {loss_inicial:.6f}")


d_E_d_pred = mse_loss_derivada(y_pred, y)

alfa = 0.05  
capa_salida.backpropagation(alpha=alfa, es_ultima=True, d_error_d_pred=d_E_d_pred)

print("\n=== COMPROBACIÓN: PASADA HACIA ATRÁS ===")
print("Gradiente del error (∂E/∂Ŷ):\n", d_E_d_pred)
print("\nDelta calculado (δ):\n", capa_salida.delta)
print("\nPesos actualizados (W_nuevo):\n", capa_salida.pesos)


y_pred_nueva = capa_salida.feed_forward(X)
loss_nuevo = mse_loss(y_pred_nueva, y)

print("\n=== VERIFICACIÓN DE APRENDIZAJE ===")
print(f"Nuevo Error (Loss) tras actualizar pesos: {loss_nuevo:.6f}")
print(f"¿El error disminuyó?: {'SÍ (El algoritmo funciona correctamente)' if loss_nuevo < loss_inicial else 'NO'}")
