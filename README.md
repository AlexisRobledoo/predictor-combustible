# ⛽ Predictor de precios de combustible — Argentina

App de **Machine Learning** que estima el precio por litro del combustible según el lugar, la marca de la estación y el tipo de combustible, entrenada con datos oficiales de la Secretaría de Energía.

🔗 **Probala online:** https://predictor-combustible-ar.streamlit.app/

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Qué hace

- 🔮 **Predice el precio** de nafta súper, premium, gasoil o GNC eligiendo provincia, localidad y marca.
- 🏷️ **Compara marcas**: muestra cuánto costaría el mismo combustible en cada bandera, en el mismo lugar.
- 🗺️ **Mapa** con los precios de las estaciones de todo el país.
- 📊 **Exploración** de precios por provincia y por marca.
- 🧠 **Transparencia**: explica cómo se entrenó el modelo, cuánto se equivoca y qué variables pesan más.

## 🧠 El modelo

| Paso | Detalle |
|---|---|
| Datos | [Precios en surtidor — Res. 314/2016](https://datos.gob.ar/dataset/energia-precios-surtidor---resolucion-3142016) (datos abiertos) |
| Limpieza | Solo precios de los últimos 60 días (por la inflación), horario diurno, sin errores de carga |
| Variables | Combustible, provincia, marca, latitud y longitud |
| Evaluación | 80% para entrenar, 20% para evaluar con datos que el modelo no vio |
| Modelos comparados | Línea base, Regresión lineal (Ridge), Random Forest, Gradient Boosting |

**Resultado:** el mejor modelo (Gradient Boosting) se equivoca en promedio unos **$31 por litro**, contra **$71** de la línea base (usar el precio mediano de cada combustible): **56% menos de error**.

> Los números exactos dependen de los datos del momento. Para verlos, corré `python modelo.py`.

## 🛠️ Tecnologías

- **Python**: pandas, scikit-learn
- **Plotly**: gráficos interactivos y mapa
- **Streamlit**: la app web

## 🚀 Cómo correrlo en tu compu

```bash
git clone https://github.com/AlexisRobledoo/predictor-combustible.git
cd predictor-combustible
pip install -r requirements.txt

python modelo.py        # entrena y muestra la comparación de modelos
streamlit run app.py    # abre la app en el navegador
```

## 📁 Estructura

```
├── app.py                 → la app web (Streamlit)
├── modelo.py              → limpieza de datos y entrenamiento
├── data/                  → CSV oficial de precios
├── requirements.txt       → librerías necesarias
└── .streamlit/config.toml → colores de la app
```

## 🔄 Actualizar los datos

Descargá el CSV nuevo desde [datos.gob.ar](https://datos.gob.ar/dataset/energia-precios-surtidor---resolucion-3142016), reemplazalo en `data/` y subilo. La app se re-entrena sola con los datos nuevos.

## ⚠️ Limitaciones

Los precios los informan las propias estaciones. Si una no actualiza, su dato puede estar desfasado. Las estimaciones son orientativas.

## 👤 Autor

**Alexis Robledo**, estudiante de la Tecnicatura en Programación (UGR, Rosario).
[Portfolio](https://alexisrobledoo.github.io) · [LinkedIn](https://linkedin.com/in/alexis-daniel-robledo) · [GitHub](https://github.com/AlexisRobledoo)
