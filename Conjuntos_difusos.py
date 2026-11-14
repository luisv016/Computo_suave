import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0. FUNCIONES DE MEMBRESÍA
# ============================================================

def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if a != b:
        idx = (x >= a) & (x <= b)
        y[idx] = (x[idx] - a) / (b - a)
    if b != c:
        idx = (x >= b) & (x <= c)
        y[idx] = (c - x[idx]) / (c - b)
    y[x == b] = 1.0
    return np.clip(y, 0, 1)

def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    y = np.ones_like(x)
    y[x < a] = 0
    y[x > d] = 0
    if b > a:
        idx = (x >= a) & (x < b)
        y[idx] = (x[idx] - a) / (b - a)
    if d > c:
        idx = (x > c) & (x <= d)
        y[idx] = (d - x[idx]) / (d - c)
    return np.clip(y, 0, 1)

def gaussmf(x, sigma, c):
    x = np.asarray(x, dtype=float)
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

def sigmf(x, a, c):
    """Sigmoidal: 1 / (1 + exp(-a*(x - c))). a>0 creciente, a<0 decreciente."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-a * (x - c)))


# ============================================================
# 1. OPERADORES DIFUSOS TIPO 1
# ============================================================

def fuzzy_union(A, B):          return np.maximum(A, B)
def fuzzy_intersection(A, B):   return np.minimum(A, B)
def fuzzy_complement(A):        return 1.0 - A




# ---- Variable 1: Temperatura (°C) — 4 términos ----
temp_x = np.linspace(-10, 45, 1001)

temperatura_mfs = {
    "Fría":         lambda x: sigmf(x, a=-0.6, c=10),          # sigmoidal decreciente
    "Templada":     lambda x: gaussmf(x, sigma=4, c=22),       # gaussiana
    "Caliente":     lambda x: trimf(x, 25, 32, 40),            # triangular
    "Muy Caliente": lambda x: trapmf(x, 35, 40, 45, 45),       # trapezoidal
}

# ---- Variable 2: Humedad relativa (%) — 3 términos ----
hum_x = np.linspace(0, 100, 1001)

humedad_mfs = {
    "Baja":  lambda x: trapmf(x, 0, 0, 20, 40),                # trapezoidal
    "Media": lambda x: gaussmf(x, sigma=12, c=55),             # gaussiana
    "Alta":  lambda x: sigmf(x, a=0.15, c=75),                 # sigmoidal creciente
}

# ---- Variable 3: Velocidad del ventilador (RPM) — 4 términos ----
vel_x = np.linspace(0, 2000, 1001)

ventilador_mfs = {
    "Baja":     lambda x: trimf(x, 0, 400, 800),               # triangular
    "Media":    lambda x: gaussmf(x, sigma=200, c=1100),       # gaussiana
    "Alta":     lambda x: trapmf(x, 1300, 1600, 2000, 2000),   # trapezoidal
    "Muy Alta": lambda x: trimf(x, 1700, 1900, 2000),          # triangular
}



fig, axs = plt.subplots(3, 1, figsize=(9, 11))

for nombre, mf in temperatura_mfs.items():
    axs[0].plot(temp_x, mf(temp_x), label=nombre)
axs[0].set_title("Variable lingüística 1: Temperatura (°C)")
axs[0].set_xlabel("Temperatura"); axs[0].set_ylabel("μ")
axs[0].legend(); axs[0].grid(True)

for nombre, mf in humedad_mfs.items():
    axs[1].plot(hum_x, mf(hum_x), label=nombre)
axs[1].set_title("Variable lingüística 2: Humedad relativa (%)")
axs[1].set_xlabel("Humedad"); axs[1].set_ylabel("μ")
axs[1].legend(); axs[1].grid(True)

for nombre, mf in ventilador_mfs.items():
    axs[2].plot(vel_x, mf(vel_x), label=nombre)
axs[2].set_title("Variable lingüística 3: Velocidad del ventilador (RPM)")
axs[2].set_xlabel("Velocidad"); axs[2].set_ylabel("μ")
axs[2].legend(); axs[2].grid(True)

plt.tight_layout()
plt.show()




A = temperatura_mfs["Templada"](temp_x)
B = temperatura_mfs["Caliente"](temp_x)

plt.figure(figsize=(9, 5))
plt.plot(temp_x, A, label="A = Templada")
plt.plot(temp_x, B, label="B = Caliente")
plt.plot(temp_x, fuzzy_union(A, B),        label="Unión  A ∪ B")
plt.plot(temp_x, fuzzy_intersection(A, B), label="Intersección  A ∩ B")
plt.plot(temp_x, fuzzy_complement(A), "--",label="Complemento  Aᶜ")
plt.title("Operadores difusos tipo 1 (sobre Temperatura)")
plt.xlabel("Temperatura (°C)"); plt.ylabel("μ")
plt.legend(); plt.grid(True)
plt.show()


# ============================================================
# 5. COMPOSICIÓN DE RELACIONES: R ∘ S
# ============================================================

def compose(R, S, tnorm="min", tconorm="max"):
    R = np.asarray(R, dtype=float)
    S = np.asarray(S, dtype=float)
    if R.shape[1] != S.shape[0]:
        raise ValueError("Dimensiones incompatibles.")
    n, k = R.shape
    m = S.shape[1]
    out = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            vals = []
            for l in range(k):
                if tnorm == "min":
                    vals.append(min(R[i, l], S[l, j]))
                elif tnorm == "product":
                    vals.append(R[i, l] * S[l, j])
            out[i, j] = max(vals) if tconorm == "max" else sum(vals)
    return out

R = np.array([[1, 0],
              [0, 1]])
S = np.array([[1, 0, 0],
              [1, 0, 1]])

print("R ∘ S  (max-min):")
print(compose(R, S, tnorm="min", tconorm="max"))
print("\nR ∘ S  (max-producto):")
print(compose(R, S, tnorm="product", tconorm="max"))




servicio_x = np.linspace(0, 10, 1001)
comida_x   = np.linspace(0, 10, 1001)
propina_y  = np.linspace(0, 30, 3001)

servicio_mfs = {
    "Malo":      lambda x: gaussmf(x, 1.5, 0),
    "Bueno":     lambda x: gaussmf(x, 1.5, 5),
    "Excelente": lambda x: gaussmf(x, 1.5, 10),
}
comida_mfs = {
    "Rancia":    lambda x: trapmf(x, 0, 0, 2, 4),
    "Deliciosa": lambda x: trapmf(x, 6, 8, 10, 10),
}
propina_mfs = {
    "Baja":         lambda y: trimf(y, 0, 5, 10),
    "Mas o Menos":  lambda y: trimf(y, 10, 15, 20),
    "Generosa":     lambda y: trimf(y, 20, 25, 30),
}

def mu1(x, mf): return float(mf(np.array([x], dtype=float))[0])

def tipper_producto(servicio_val, comida_val):
    # Fuerzas de disparo
    alpha1 = max(mu1(servicio_val, servicio_mfs["Malo"]),
                 mu1(comida_val,   comida_mfs["Rancia"]))
    alpha2 = mu1(servicio_val, servicio_mfs["Bueno"])
    alpha3 = max(mu1(servicio_val, servicio_mfs["Excelente"]),
                 mu1(comida_val,   comida_mfs["Deliciosa"]))

    # Implicación Mamdani con T-Norma = Producto
    s1 = alpha1 * propina_mfs["Baja"](propina_y)
    s2 = alpha2 * propina_mfs["Mas o Menos"](propina_y)
    s3 = alpha3 * propina_mfs["Generosa"](propina_y)

    agregada = np.maximum.reduce([s1, s2, s3])

    if np.sum(agregada) == 0:
        prop = 0.0
    else:
        prop = np.sum(propina_y * agregada) / np.sum(agregada)

    return alpha1, alpha2, alpha3, s1, s2, s3, agregada, prop


casos = [(1, 9), (10, 3), (5, 5)]
print("\nResultados Tipper con T-Norma = Producto:")
for s, c in casos:
    a1, a2, a3, *_ , prop = tipper_producto(s, c)
    print(f"Servicio={s}, Comida={c}  ->  Propina = {prop:.2f}%   "
          f"(α1={a1:.3f}, α2={a2:.3f}, α3={a3:.3f})")

# Visualización de la agregación para el caso (1,9)
a1, a2, a3, s1, s2, s3, agregada, prop = tipper_producto(1, 9)
plt.figure(figsize=(10, 5))
plt.plot(propina_y, s1, label="R1 (Baja)")
plt.plot(propina_y, s2, label="R2 (Mas o Menos)")
plt.plot(propina_y, s3, label="R3 (Generosa)")
plt.plot(propina_y, agregada, "k--", lw=2, label="Agregación")
plt.axvline(prop, color="red", ls=":", label=f"Centroide = {prop:.2f}%")
plt.title("Tipper con T-Norma = Producto (Servicio=1, Comida=9)")
plt.xlabel("Propina (%)"); plt.ylabel("μ")
plt.legend(); plt.grid(True)
plt.show()