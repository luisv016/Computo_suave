import numpy as np
import csv  # Librería estándar para abrir archivos CSV

# ---------------- Activaciones ----------------
def ReLu(x):
    return np.maximum(0, x)

def ReLu_deriv(x):
    return (x > 0).astype(float)

def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoide_deriv(x):
    s = sigmoide(x)
    return s * (1.0 - s)

DERIVADAS = {ReLu: ReLu_deriv, sigmoide: sigmoide_deriv}

# ---------------- Pérdida ----------------
def mse(T, Yhat):
    return np.mean(0.5 * (T - Yhat) ** 2)

def mse_deriv(T, Yhat):
    return -(T - Yhat)

# ---------------- Capa ----------------
class Capa:
    def __init__(self, entradas, neuronas, activacion, bias=False, pesos=None):
        self.activacion = activacion
        self.activacion_deriv = DERIVADAS[activacion]
        self.bias = bias
        if pesos is not None:
            self.pesos = np.array(pesos, dtype=float)
        else:
            filas = entradas + 1 if bias else entradas
           
            self.pesos = (np.random.rand(filas, neuronas) - 0.5) * np.sqrt(2.0 / entradas)

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

# ---------------- Red ----------------
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

    def train(self, X, T, epocas, tol, alpha, cada=500, X_val=None, T_val=None):
       
        for epoca in range(1, epocas + 1):
           
            YH = self.feedforward(X)
            loss = mse(T, YH)
            
            
            if epoca % cada == 0 or epoca == 1:
                if X_val is not None:
                   
                    estados = [(c.entrada_aum, c.N, c.salida) for c in self.capas]
                    
                   
                    loss_val = mse(T_val, self.feedforward(X_val))
                    
                    
                    for c, (ea, N, s) in zip(self.capas, estados):
                        c.entrada_aum, c.N, c.salida = ea, N, s
                    
                    print(f"Epoca {epoca} | MSE train: {loss:.6f} | MSE val: {loss_val:.6f}")
                else:
                    print(f"Epoca {epoca} | Pérdida (MSE): {loss:.6f}")
            
            if loss <= tol:
                print(f"Epoca {epoca} | Pérdida (MSE): {loss:.6f} (tolerancia alcanzada)")
                return
            
           
            self.backpropagation(T, alpha)
        
        print(f"Epoca {epocas} | Pérdida (MSE) Final: {loss:.6f}")

    def predecir(self, X):
        return self.feedforward(X)

# ---------------- Apertura y Lectura del Archivo CSV ----------------
def cargar_desde_archivo_csv(nombre_archivo):
    X_lista = []
    T_lista = []
    
    with open(nombre_archivo, mode='r', encoding='utf-8') as f:
        lector = csv.reader(f)
        next(lector)  # Ignorar la línea de encabezados
        
        for fila in lector:
            if not fila: continue 
            
            # 1. One-hot encoding para la columna categórica 'Sex' (M, F, I)
            sexo = fila[0]
            if sexo == 'M':   encoded_sex = [1.0, 0.0, 0.0]
            elif sexo == 'F': encoded_sex = [0.0, 1.0, 0.0]
            else:             encoded_sex = [0.0, 0.0, 1.0] # Infante
            
            # 2. Guardar las 7 características numéricas físicas (Length a Shell weight)
            caracteristicas_fisicas = [float(valor) for valor in fila[1:8]]
            
            # 3. Guardar la columna objetivo de anillos
            anillos = [float(fila[8])]
            
            X_lista.append(encoded_sex + caracteristicas_fisicas)
            T_lista.append(anillos)
            
    X = np.array(X_lista, dtype=float)
    T = np.array(T_lista, dtype=float)
    
    # Conservamos límites originales para desnormalizar al final
    t_min, t_max = T.min(), T.max()
    
    # Normalización Min-Max (Obligatoria para la estabilidad de la función Sigmoide)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_max_minus_min = np.where(X_max - X_min == 0, 1, X_max - X_min)
    X_norm = (X - X_min) / X_max_minus_min
    
    T_norm = (T - t_min) / (t_max - t_min) if (t_max - t_min) != 0 else T
    
    return X_norm, T_norm, t_min, t_max

# ---------------- Bloque de Ejecución Principal ----------------
if __name__ == "__main__":
    np.random.seed(42)  # Controlar la aleatoriedad inicial
    
    
    archivo_csv = "abalone.csv" 
    
    try:
        X, T, t_min, t_max = cargar_desde_archivo_csv(archivo_csv)
        print(f"Éxito: Archivo '{archivo_csv}' cargado correctamente.")
        print(f"Estructura matemática -> Entradas (X): {X.shape} | Objetivo (T): {T.shape}")
        
      
        n_total = X.shape[0]
        indices = np.random.permutation(n_total)  # Mezclar aleatoriamente
        corte = int(0.6 * n_total)                # 60% para entrenamiento
        
        idx_train = indices[:corte]
        idx_test  = indices[corte:]
        
        X_train, T_train = X[idx_train], T[idx_train]
        X_test,  T_test  = X[idx_test],  T[idx_test]
        
        print(f"\nDivisión de datos:")
        print(f"  Entrenamiento (60%): {X_train.shape[0]} muestras")
        print(f"  Test          (40%): {X_test.shape[0]}  muestras")
        # ============================================================
        
        # Estructura: 10 entradas (3 One-hot + 7 físicas), 8 neuronas ocultas, 1 neurona de salida (Rings)
        red = Red([
            Capa(entradas=10, neuronas=8, activacion=sigmoide, bias=True),
            Capa(entradas=8, neuronas=1, activacion=sigmoide, bias=True)
        ])
        
        # Entrenamos con X_train, T_train y monitoreamos con el conjunto de test
        red.train(X_train, T_train, epocas=12000, tol=1e-5, alpha=0.3, cada=3000,
                  X_val=X_test, T_val=T_test)
        
        # ============================================================
        # Evaluación sobre ENTRENAMIENTO
        # ============================================================
        pred_train_norm = red.predecir(X_train)
        pred_train_real = pred_train_norm * (t_max - t_min) + t_min
        T_train_real    = T_train * (t_max - t_min) + t_min
        
        mse_train = np.mean((T_train_real - pred_train_real) ** 2)
        mae_train = np.mean(np.abs(T_train_real - pred_train_real))
        rmse_train = np.sqrt(mse_train)
        
        # ============================================================
        # Evaluación sobre TEST (datos nunca vistos)
        # ============================================================
        pred_test_norm = red.predecir(X_test)
        pred_test_real = pred_test_norm * (t_max - t_min) + t_min
        T_test_real    = T_test * (t_max - t_min) + t_min
        
        mse_test  = np.mean((T_test_real - pred_test_real) ** 2)
        mae_test  = np.mean(np.abs(T_test_real - pred_test_real))
        rmse_test = np.sqrt(mse_test)
        
        print("\n" + "="*55)
        print("MÉTRICAS FINALES (en anillos reales)")
        print("="*55)
        print(f"  ENTRENAMIENTO (60%):")
        print(f"    MSE  = {mse_train:.4f}")
        print(f"    RMSE = {rmse_train:.4f} anillos")
        print(f"    MAE  = {mae_train:.4f} anillos")
        print(f"\n  TEST (40%):")
        print(f"    MSE  = {mse_test:.4f}")
        print(f"    RMSE = {rmse_test:.4f} anillos")
        print(f"    MAE  = {mae_test:.4f} anillos")
        
        # Diagnóstico de sobreajuste
        print("\n  Diagnóstico:")
        if mse_test > 1.5 * mse_train:
            print("    ⚠ Posible sobreajuste (MSE test >> MSE train)")
        elif mse_test < 0.7 * mse_train:
            print("    ⚠ Posible subajuste (MSE test << MSE train, poco común)")
        else:
            print("    ✓ Generalización razonable (MSE train ≈ MSE test)")
        
        # ============================================================
        # Muestra de resultados sobre TEST
        # ============================================================
        print("\nMuestra de resultados sobre TEST (Primeras 10 filas):")
        print("Valor Real (Anillos) | Predicción de la Red | Error")
        print("-" * 55)
        for r, p in zip(T_test_real[:10], pred_test_real[:10]):
            error = p[0] - r[0]
            print(f"     {int(r[0]):2d}              |       {p[0]:.2f}        |  {error:+.2f}")
            
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{archivo_csv}'. Revisa que esté guardado en el mismo directorio.")