"""
⛽ Predictor de precios de combustible — app web con Streamlit.

Para correrla en tu compu:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from modelo import entrenar

st.set_page_config(page_title="Predictor de combustible AR", page_icon="⛽", layout="wide")

ACENTO = "#7c6cff"
ESCALA = ["#d9d4ff", "#a99dff", "#7c6cff", "#5442d6", "#2e2190"]  # un solo tono, de claro a oscuro


# ---------------------------------------------------------------------------
# Entrenamos UNA sola vez: Streamlit guarda el resultado en caché
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Entrenando el modelo con los datos oficiales…")
def cargar():
    return entrenar()


r = cargar()
modelo, df = r["modelo"], r["datos"]
pesos = lambda v: f"$ {v:,.0f}".replace(",", ".")
pesos_md = lambda v: pesos(v).replace("$", "\\$")  # en markdown, $ se confunde con fórmulas
miles = lambda v: f"{v:,}".replace(",", ".")


def estilo(fig, alto=380):
    """Estilo común para todos los gráficos."""
    fig.update_layout(
        height=alto,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


# ---------------------------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------------------------
st.title("⛽ Predictor de precios de combustible")
st.markdown(
    f"Modelo de **Machine Learning** entrenado con **{miles(len(df))} precios reales** de estaciones de "
    f"servicio de toda Argentina, informados a la Secretaría de Energía hasta el "
    f"**{r['ultima_fecha']:%d/%m/%Y}**."
)

tab_pred, tab_mapa, tab_datos, tab_modelo = st.tabs(
    ["🔮 Predecir", "🗺️ Mapa", "📊 Explorar datos", "🧠 Cómo funciona"]
)

# ---------------------------------------------------------------------------
# 1) PREDECIR
# ---------------------------------------------------------------------------
with tab_pred:
    st.subheader("¿Cuánto debería costar el combustible?")
    st.caption("Elegí dónde, qué marca y qué combustible. El modelo estima el precio por litro.")

    c1, c2, c3, c4 = st.columns(4)
    productos = sorted(df["producto"].unique())
    producto = c1.selectbox("Combustible", productos, index=productos.index("Nafta súper"))

    provincias = sorted(df["provincia"].unique())
    provincia = c2.selectbox("Provincia", provincias, index=provincias.index("Santa Fe"))

    localidades = sorted(df.loc[df["provincia"] == provincia, "localidad"].unique())
    pos = localidades.index("Rosario") if "Rosario" in localidades else 0
    localidad = c3.selectbox("Localidad", localidades, index=pos)

    banderas = sorted(df["bandera"].unique())
    bandera = c4.selectbox("Marca de la estación", banderas,
                           index=banderas.index("YPF") if "YPF" in banderas else 0)

    # La ubicación la tomamos del centro de las estaciones de esa localidad
    lugar = df[(df["provincia"] == provincia) & (df["localidad"] == localidad)]
    lat, lon = lugar["latitud"].mean(), lugar["longitud"].mean()

    entrada = pd.DataFrame([{"producto": producto, "provincia": provincia, "bandera": bandera,
                             "latitud": lat, "longitud": lon}])
    prediccion = modelo.predict(entrada)[0]
    error = r["error_producto"].get(producto, 40)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Precio estimado por litro", pesos(prediccion))
    m2.metric("Margen de error típico", f"± {pesos(error)}")

    reales = lugar[lugar["producto"] == producto]
    if len(reales):
        promedio = reales["precio"].mean()
        dif = prediccion - promedio
        m3.metric(f"Promedio real en {localidad}", pesos(promedio),
                  help=f"Promedio de las {len(reales)} estaciones de {localidad} (todas las marcas).")
    else:
        m3.metric(f"Promedio real en {localidad}", "Sin datos",
                  help="No hay estaciones de esa localidad con este combustible en los datos.")

    # ¿Cuánto cambia según la marca? Predecimos lo mismo para cada bandera
    st.markdown("##### ¿Y si cargás en otra marca, en el mismo lugar?")
    escenarios = entrada.loc[entrada.index.repeat(len(banderas))].assign(bandera=banderas)
    escenarios["precio"] = modelo.predict(escenarios)
    escenarios = escenarios.sort_values("precio")
    escenarios["es_elegida"] = escenarios["bandera"] == bandera

    # Usamos puntos (no barras) porque el eje no empieza en cero: así no exageramos diferencias
    fig = px.scatter(escenarios, x="precio", y="bandera",
                     color="es_elegida", color_discrete_map={True: ACENTO, False: "#9a97ad"},
                     category_orders={"bandera": escenarios["bandera"].tolist()},
                     text=escenarios["precio"].map(pesos))
    fig.update_traces(marker=dict(size=14), textposition="middle right",
                      hovertemplate="%{y}: %{text}<extra></extra>")
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(range=[escenarios["precio"].min() - 40, escenarios["precio"].max() + 70],
                     showgrid=True, gridcolor="rgba(128,128,128,0.15)", tickprefix="$ ")
    st.plotly_chart(estilo(fig, 60 + 34 * len(banderas)), use_container_width=True)

    pocos = df[(df["bandera"] == bandera) & (df["producto"] == producto)]
    if len(pocos) < 30:
        st.caption(f"⚠️ Hay pocos precios recientes de {bandera} para {producto} ({len(pocos)}), "
                   "así que esta estimación es menos precisa.")

    if len(reales):
        with st.expander(f"Ver las {len(reales)} estaciones reales de {localidad} ({producto})"):
            tabla = (reales.sort_values("precio")
                     [["bandera", "empresa", "direccion", "precio", "fecha_vigencia"]]
                     .rename(columns={"bandera": "Marca", "empresa": "Empresa",
                                      "direccion": "Dirección", "precio": "Precio",
                                      "fecha_vigencia": "Actualizado"}))
            st.dataframe(tabla, hide_index=True, use_container_width=True,
                         column_config={
                             "Precio": st.column_config.NumberColumn(format="$ %d"),
                             "Actualizado": st.column_config.DateColumn(format="DD/MM/YYYY"),
                         })

# ---------------------------------------------------------------------------
# 2) MAPA
# ---------------------------------------------------------------------------
with tab_mapa:
    st.subheader("Precios en todo el país")
    prod_mapa = st.selectbox("Combustible ", productos, index=productos.index("Nafta súper"),
                             key="prod_mapa")
    datos_mapa = df[df["producto"] == prod_mapa]

    fig = px.scatter_map(
        datos_mapa, lat="latitud", lon="longitud", color="precio",
        color_continuous_scale=ESCALA, zoom=3.4, center=dict(lat=-38.5, lon=-63.5),
        hover_name="empresa",
        hover_data={"bandera": True, "localidad": True, "precio": ":$,.0f",
                    "latitud": False, "longitud": False},
        labels={"precio": "Precio", "bandera": "Marca", "localidad": "Localidad"},
        map_style="carto-positron",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85))
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cada punto es una estación. Más oscuro = más caro.")

# ---------------------------------------------------------------------------
# 3) EXPLORAR DATOS
# ---------------------------------------------------------------------------
with tab_datos:
    prod_exp = st.selectbox("Combustible  ", productos, index=productos.index("Nafta súper"),
                            key="prod_exp")
    d = df[df["producto"] == prod_exp]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estaciones", f"{len(d):,}".replace(",", "."))
    k2.metric("Precio mediano", pesos(d["precio"].median()))
    k3.metric("Más barato", pesos(d["precio"].min()))
    k4.metric("Más caro", pesos(d["precio"].max()))

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### Precio mediano por provincia")
        por_prov = d.groupby("provincia")["precio"].median().sort_values().reset_index()
        fig = px.scatter(por_prov, x="precio", y="provincia", color_discrete_sequence=[ACENTO])
        fig.update_traces(marker=dict(size=11),
                          hovertemplate="%{y}: $%{x:,.0f}<extra></extra>")
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", tickprefix="$ ")
        st.plotly_chart(estilo(fig, 620), use_container_width=True)
        st.caption("La Patagonia es más barata porque paga menos impuestos a los combustibles.")

    with g2:
        st.markdown("##### Precio según la marca")
        orden = d.groupby("bandera")["precio"].median().sort_values().index.tolist()
        fig = px.box(d, x="precio", y="bandera", category_orders={"bandera": orden},
                     color_discrete_sequence=[ACENTO], points=False)
        fig.update_layout(xaxis_title="Precio por litro", yaxis_title=None)
        st.plotly_chart(estilo(fig, 620), use_container_width=True)
        st.caption("La caja muestra dónde está la mitad de los precios; la línea del medio es la mediana.")

# ---------------------------------------------------------------------------
# 4) CÓMO FUNCIONA
# ---------------------------------------------------------------------------
with tab_modelo:
    st.subheader("Cómo funciona el modelo")
    st.markdown(f"""
1. **Datos**: precios en surtidor que las estaciones informan a la Secretaría de Energía
   ([datos abiertos](https://datos.gob.ar/dataset/energia-precios-surtidor---resolucion-3142016)).
2. **Limpieza**: me quedo con precios actualizados en los últimos 60 días (por la inflación, los
   precios viejos confunden al modelo), uso solo horario diurno y saco errores de carga.
3. **Variables**: combustible, provincia, marca de la estación y ubicación (latitud y longitud).
4. **Entrenamiento**: separo los datos en **{miles(r['n_train'])} para entrenar** y
   **{miles(r['n_test'])} para evaluar**. Así mido el error con precios que el modelo nunca vio.
5. **Comparación**: pruebo 4 modelos y me quedo con el que menos se equivoca: **{r['nombre']}**.
""")

    t1, t2 = st.columns([1.2, 1])
    with t1:
        st.markdown("##### Comparación de modelos")
        st.dataframe(
            r["tabla"], hide_index=True, use_container_width=True,
            column_config={
                "Error promedio ($)": st.column_config.NumberColumn(format="$ %.0f"),
                "R²": st.column_config.NumberColumn(format="%.3f"),
            })
        base = r["tabla"].iloc[-1]["Error promedio ($)"]
        mejor = r["tabla"].iloc[0]["Error promedio ($)"]
        st.markdown(
            f"La **línea base** solo dice \"el precio mediano de ese combustible\" y se equivoca en "
            f"promedio **{pesos_md(base)}**. El mejor modelo baja ese error a "
            f"**{pesos_md(mejor)}** "
            f"(**{(1 - mejor / base):.0%} menos**)."
        )
        st.caption("Error promedio: cuánto se aleja la predicción del precio real, en pesos. "
                   "R²: qué parte de las diferencias de precio explica el modelo (1 = todo).")

    with t2:
        st.markdown("##### ¿Qué pesa más en el precio?")
        imp = r["importancia"].copy()
        imp["Variable"] = imp["Variable"].replace({
            "producto": "Combustible", "bandera": "Marca", "provincia": "Provincia",
            "latitud": "Latitud (norte–sur)", "longitud": "Longitud (este–oeste)"})
        imp = imp.sort_values("Importancia")
        fig = px.bar(imp, x="Importancia", y="Variable", orientation="h",
                     color_discrete_sequence=[ACENTO])
        fig.update_traces(hovertemplate="%{y}: %{x:.1f}<extra></extra>")
        fig.update_layout(xaxis_title="Aumento del error sin esa variable ($)", yaxis_title=None)
        st.plotly_chart(estilo(fig, 300), use_container_width=True)

    st.info(
        "**Limitaciones:** el modelo aprende de los precios informados por las estaciones; si una "
        "estación no actualiza, su dato puede estar desfasado. Las estimaciones son orientativas.",
        icon="ℹ️",
    )

# ---------------------------------------------------------------------------
# BARRA LATERAL
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⛽ Sobre el proyecto")
    st.markdown(
        "Proyecto de Machine Learning hecho por **Alexis Robledo**, estudiante de la "
        "Tecnicatura en Programación (UGR, Rosario)."
    )
    st.markdown(
        "- 🌐 [Mi portfolio](https://alexisrobledoo.github.io)\n"
        "- 💻 [Código en GitHub](https://github.com/AlexisRobledoo/predictor-combustible)\n"
        "- 💼 [LinkedIn](https://linkedin.com/in/alexis-daniel-robledo)"
    )
    st.caption("Python · pandas · scikit-learn · Plotly · Streamlit")
