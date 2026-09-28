import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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
            self.pesos = np.random.rand(filas, neuronas) - 0.5

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


if __name__ == "__main__":
    np.random.seed(0)
    
    
    nombre_archivo = "phishing_site_urls.csv" 
    df = pd.read_csv(nombre_archivo)
    
    print(f"Dataset original detectado con {len(df)} URLs.")
    
  
    df_muestra = df.sample(n=15000, random_state=42).reset_index(drop=True)
    print(f"Trabajando con una muestra optimizada de {len(df_muestra)} URLs para agilizar NumPy.")
    
    # 2. Convertir etiquetas de texto ('bad' -> 1.0, 'good' -> 0.0)
    df_muestra['Target'] = df_muestra['Label'].apply(lambda x: 1.0 if x == 'bad' else 0.0)
    
    # 3. Vectorizar el texto de las URLs (Subimos a 100 características para mayor precisión)
    vectorizador = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=100)
    X_numerico = vectorizador.fit_transform(df_muestra['URL']).toarray()
    T_numerico = df_muestra['Target'].values.reshape(-1, 1)
    
    # 4. Dividir el dataset (80% entrenamiento, 20% prueba)
    X_train, X_test, T_train, T_test = train_test_split(X_numerico, T_numerico, test_size=0.2, random_state=42)
    
    # Obtener el número de entradas de forma dinámica
    num_entradas = X_train.shape[1]
    print(f"La red recibirá {num_entradas} entradas numéricas por cada URL de la muestra.\n")
    
    # 5. Inicializar la red (Aumentamos un poco las neuronas ocultas debido a las 100 entradas)
    red = Red([
        Capa(entradas=num_entradas, neuronas=15, activacion=sigmoide, bias=True),
        Capa(entradas=15, neuronas=1, activacion=sigmoide, bias=True), # 1 salida (Cerca de 1 = Phishing, Cerca de 0 = Seguro)
    ])
    
    
    red.train(X_train, T_train, epocas=3000, tol=1e-4, alpha=0.5, cada=500)
    
    
    print("\n--- Evaluación con una muestra de prueba (URLs no vistas en entrenamiento) ---")
    predicciones = red.predecir(X_test)
    
    aciertos = 0
    for i in range(len(X_test)):
        valor_real = T_test[i][0]
        score = predicciones[i][0]
        predicho = 1.0 if score >= 0.5 else 0.0
        if predicho == valor_real:
            aciertos += 1
            
        if i < 10: # Solo imprimimos los primeros 10 para no saturar la pantalla
            txt_real = "MALICIOSA (bad)" if valor_real == 1.0 else "SEGURA (good)"
            txt_predicho = "MALICIOSA (bad)" if predicho == 1.0 else "SEGURA (good)"
            print(f"URL {i+1} | Real: {txt_real} | Predicción: {txt_predicho} (Score: {score:.3f})")
            
    print(f"\nExactitud (Accuracy) en el conjunto de prueba: {(aciertos / len(X_test)) * 100:.2f}%")

    print("\n" + "="*50)
    print("PROBAR UNA URL PERSONALIZADA")
    print("="*50)
    
    
    mi_url_prueba = "https://wikipedia.org"
    
    
    mi_url_numerica = vectorizador.transform([mi_url_prueba]).toarray()
    

    prediccion_personalizada = red.predecir(mi_url_numerica)
    score_personalizado = prediccion_personalizada[0][0] 
    
    
    print(f"URL introducida: {mi_url_prueba}")
    print(f"Score de sospecha: {score_personalizado:.3f}")
    
    if score_personalizado >= 0.5:
        print("VEREDICTO: ¡ALERTA! La red detectó que esta URL es MALICIOSA (bad) 🚨")
    else:
        print("VEREDICTO: La red detectó que esta URL es SEGURA (good) ")

