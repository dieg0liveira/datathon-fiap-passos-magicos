import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

FEATURE_LABELS = {
    "IAA": "Desempenho acadêmico geral",
    "IPS": "Situação psicossocial",
    "IPP": "Continuidade e permanência nos estudos",
    "IPV": "Situação de vulnerabilidade",
    "IAN": "Necessidade de apoio"
}

FEATURES = ["IAA", "IPS", "IPP", "IPV", "IAN"]

st.set_page_config(page_title="Predição - Ponto de Virada", layout="wide")

st.title("🔮 Predição - Ponto de Virada")

st.markdown("""
Nesta página, você pode informar os indicadores do aluno para estimar a probabilidade de ele atingir um **ponto de virada** em sua trajetória educacional.

O modelo utiliza dados de desempenho, contexto psicossocial, permanência nos estudos, vulnerabilidade e necessidade de apoio para gerar essa previsão.
""")

@st.cache_resource
def load_model():
    model = joblib.load("random_forest_ponto_virada_model.pkl")
    scaler = joblib.load("random_forest_ponto_virada_scaler.pkl")
    return model, scaler

@st.cache_data
def load_feature_importance():
    return pd.read_csv("output/ponto_virada_feature_importance.csv")

model, scaler = load_model()
feature_importance_df = load_feature_importance()

st.subheader("Informe os indicadores do aluno")

st.info("""
Use a escala de 0 a 10:

- **0 a 3 → Baixo**
- **4 a 7 → Médio**
- **8 a 10 → Alto**

Quanto maior o valor, maior a intensidade daquele indicador.
""")

c1, c2, c3 = st.columns(3)

with c1:
    iaa = st.slider(
        FEATURE_LABELS["IAA"], 0.0, 10.0, 5.0, 0.1,
        help="Avalia o desempenho escolar geral do aluno."
    )
    ips = st.slider(
        FEATURE_LABELS["IPS"], 0.0, 10.0, 5.0, 0.1,
        help="Representa aspectos emocionais, sociais e de contexto do aluno."
    )

with c2:
    ipp = st.slider(
        FEATURE_LABELS["IPP"], 0.0, 10.0, 5.0, 0.1,
        help="Relaciona-se à continuidade e permanência do aluno nos estudos."
    )
    ipv = st.slider(
        FEATURE_LABELS["IPV"], 0.0, 10.0, 5.0, 0.1,
        help="Indica o nível de vulnerabilidade social do aluno."
    )

with c3:
    ian = st.slider(
        FEATURE_LABELS["IAN"], 0.0, 10.0, 5.0, 0.1,
        help="Mostra o quanto o aluno precisa de apoio adicional."
    )

if st.button("Prever ponto de virada"):
    entrada = pd.DataFrame([{
        "IAA": iaa,
        "IPS": ips,
        "IPP": ipp,
        "IPV": ipv,
        "IAN": ian,
    }])

    if hasattr(scaler, "feature_names_in_"):
        entrada = entrada[scaler.feature_names_in_]
    else:
        entrada = entrada[FEATURES]

    entrada_scaled = scaler.transform(entrada)
    pred = model.predict(entrada_scaled)[0]

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(entrada_scaled)[0][1])
        st.metric("Probabilidade estimada de atingir o ponto de virada", f"{prob:.2%}")

    if pred == 1:
        st.success("Perfil com maior probabilidade de atingir o ponto de virada.")
    else:
        st.error("Perfil com menor probabilidade de atingir o ponto de virada.")

st.divider()
st.subheader("Importância dos indicadores no modelo")

# Tradução das features no gráfico
feature_importance_df["Feature_Traduzida"] = feature_importance_df["Feature"].map(FEATURE_LABELS)

# Ordenação por importância
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(
    feature_importance_df["Feature_Traduzida"],
    feature_importance_df["Importance"]
)
ax.set_xlabel("Importância")
ax.set_ylabel("Indicador")
ax.set_title("Importância dos Indicadores - Random Forest")
ax.invert_yaxis()
st.pyplot(fig, clear_figure=True)

with st.expander("Ver tabela de importância"):
    st.dataframe(feature_importance_df, use_container_width=True)