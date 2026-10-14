"""
Predictor de precios de combustible — limpieza de datos y entrenamiento.

Se puede usar de dos formas:
  - Desde la app (app.py), que llama a `entrenar()`.
  - Solo, desde la terminal, para ver cómo le va a cada modelo:
        python modelo.py
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

RUTA_DATOS = "data/precios-en-surtidor-resolucin-3142016.csv"

# Solo usamos precios actualizados en los últimos N días.
# El archivo trae el ÚLTIMO precio que informó cada estación, y algunas no
# actualizan hace meses: con la inflación, esos precios viejos confundirían al modelo.
DIAS_RECIENTES = 60

# Nombres más cortos para mostrar en la app
NOMBRES_PRODUCTO = {
    "Nafta (súper) entre 92 y 95 Ron": "Nafta súper",
    "Nafta (premium) de más de 95 Ron": "Nafta premium",
    "Gas Oil Grado 2": "Gasoil común",
    "Gas Oil Grado 3": "Gasoil premium",
    "GNC": "GNC",
}

NOMBRES_BANDERA = {
    "SHELL C.A.P.S.A.": "SHELL",
    "DAPSA S.A.": "DAPSA",
    "SIN EMPRESA BANDERA": "SIN BANDERA",
}

CATEGORICAS = ["producto", "provincia", "bandera"]
NUMERICAS = ["latitud", "longitud"]
COLUMNAS_MODELO = CATEGORICAS + NUMERICAS


# ---------------------------------------------------------------------------
# 1) LIMPIEZA
# ---------------------------------------------------------------------------
def cargar_datos(ruta=RUTA_DATOS, dias=DIAS_RECIENTES):
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    df["fecha_vigencia"] = pd.to_datetime(df["fecha_vigencia"], errors="coerce")

    # a) Nos quedamos con los precios recientes
    ultima = df["fecha_vigencia"].max()
    df = df[df["fecha_vigencia"] >= ultima - pd.Timedelta(days=dias)]

    # b) Diurno y nocturno casi siempre valen lo mismo: usamos solo diurno
    df = df[df["tipohorario"] == "Diurno"]

    # c) Sacamos filas sin ubicación
    df = df.dropna(subset=["latitud", "longitud", "precio"])

    # d) Nombres más prolijos
    df["producto"] = df["producto"].map(NOMBRES_PRODUCTO).fillna(df["producto"])
    df["bandera"] = df["empresabandera"].replace(NOMBRES_BANDERA)
    df["provincia"] = df["provincia"].str.title().replace({"Capital Federal": "CABA"})
    df["localidad"] = df["localidad"].str.title()

    # e) Sacamos precios absurdos (errores de carga): lejos de la mediana de su producto
    mediana = df.groupby("producto")["precio"].transform("median")
    df = df[(df["precio"] > mediana * 0.6) & (df["precio"] < mediana * 1.4)]

    columnas = ["empresa", "direccion", "localidad", "provincia", "bandera",
                "producto", "precio", "fecha_vigencia", "latitud", "longitud"]
    return df[columnas].reset_index(drop=True), ultima


# ---------------------------------------------------------------------------
# 2) MODELOS
# ---------------------------------------------------------------------------
class PromedioPorProducto(BaseEstimator, RegressorMixin):
    """Línea base honesta: predice el precio mediano de cada combustible.
    Si un modelo no le gana a esto, no sirve para nada."""

    def fit(self, X, y):
        self.medianas_ = y.groupby(X["producto"]).median()
        self.general_ = y.median()
        return self

    def predict(self, X):
        return X["producto"].map(self.medianas_).fillna(self.general_).to_numpy()


def crear_modelos():
    """Devuelve los modelos que vamos a comparar."""
    # Los modelos lineales necesitan las categorías en "one-hot" (una columna por valor)
    onehot = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ("num", "passthrough", NUMERICAS),
    ])
    # Los árboles se arreglan con un número por categoría
    ordinal = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAS),
        ("num", "passthrough", NUMERICAS),
    ])

    return {
        "Línea base: mediana por combustible": PromedioPorProducto(),
        "Regresión lineal (Ridge)": Pipeline([("prep", onehot), ("modelo", Ridge(alpha=1.0))]),
        "Random Forest": Pipeline([("prep", ordinal), ("modelo", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("prep", ordinal), ("modelo", HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, categorical_features=[0, 1, 2], random_state=42))]),
    }


def entrenar(ruta=RUTA_DATOS):
    """Limpia los datos, compara modelos y devuelve el mejor ya entrenado."""
    df, ultima = cargar_datos(ruta)
    X, y = df[COLUMNAS_MODELO], df["precio"]

    # 80% para entrenar, 20% para evaluar con datos que el modelo nunca vio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["producto"])

    resultados = []
    entrenados = {}
    for nombre, modelo in crear_modelos().items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        resultados.append({
            "Modelo": nombre,
            "Error promedio ($)": mean_absolute_error(y_test, pred),
            "R²": r2_score(y_test, pred),
        })
        entrenados[nombre] = modelo

    tabla = pd.DataFrame(resultados).sort_values("Error promedio ($)").reset_index(drop=True)
    mejor_nombre = tabla.iloc[0]["Modelo"]
    mejor = entrenados[mejor_nombre]

    # ¿Qué variable pesa más? Desordenamos cada columna y medimos cuánto empeora el modelo
    imp = permutation_importance(mejor, X_test, y_test, n_repeats=8, random_state=42,
                                 scoring="neg_mean_absolute_error")
    importancia = (pd.DataFrame({"Variable": COLUMNAS_MODELO, "Importancia": imp.importances_mean})
                   .sort_values("Importancia", ascending=False))

    # Error del mejor modelo por producto (para mostrar un rango en la app)
    pred = mejor.predict(X_test)
    error_producto = (pd.DataFrame({"producto": X_test["producto"], "err": (y_test - pred).abs()})
                      .groupby("producto")["err"].mean().to_dict())

    # Lo re-entrenamos con TODOS los datos para usarlo en la app
    mejor.fit(X, y)

    return {
        "modelo": mejor,
        "nombre": mejor_nombre,
        "tabla": tabla,
        "importancia": importancia,
        "error_producto": error_producto,
        "datos": df,
        "ultima_fecha": ultima,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ---------------------------------------------------------------------------
# Si lo corrés directo: python modelo.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    r = entrenar()
    print(f"Datos usados: {len(r['datos'])} precios (hasta {r['ultima_fecha']:%d/%m/%Y})")
    print(f"Entrenamiento: {r['n_train']} · Prueba: {r['n_test']}\n")
    print(r["tabla"].round(3).to_string(index=False))
    print(f"\nMejor modelo: {r['nombre']}\n")
    print("Importancia de cada variable:")
    print(r["importancia"].round(2).to_string(index=False))
