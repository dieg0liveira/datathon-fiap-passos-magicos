import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

FEATURE_LABELS = {
    "IAA": "Desempenho acadêmico geral",
    "IEG": "Engajamento nas atividades",
    "IPS": "Situação psicossocial",
    "IDA": "Desenvolvimento acadêmico",
    "IPP": "Continuidade e permanência nos estudos",
    "IPV": "Situação de vulnerabilidade",
    "IAN": "Necessidade de apoio",
    "INDE": "Nível de desenvolvimento educacional"
}

st.set_page_config(page_title="Predição - Bolsa", layout="wide")

st.title("🎓 Predição - Indicação de Bolsa")

st.markdown("""
Nesta página, você pode informar os indicadores do aluno para estimar a probabilidade de ele ser **indicado para receber bolsa de estudos**.

O modelo considera fatores como desempenho acadêmico, engajamento, contexto social, permanência nos estudos e necessidade de apoio para apoiar essa previsão.
""")

FEATURES = ["IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN", "INDE"]

@st.cache_resource
def load_assets():
    model = joblib.load("perfil_receber_bolsa_estudos_model.pkl")
    scaler = joblib.load("perfil_receber_bolsa_estudos_scaler.pkl")
    return model, scaler

@st.cache_data
def load_training_history():
    return pd.read_csv("model_training_data_hist.csv", sep=";")

model, scaler = load_assets()
hist = load_training_history()

st.subheader("Informe os indicadores do aluno")

st.info("""
Use a escala de 0 a 10:

- **0 a 3 → Baixo**
- **4 a 7 → Médio**
- **8 a 10 → Alto**

Quanto maior o valor, maior a intensidade daquele indicador.
""")

c1, c2, c3, c4 = st.columns(4)

with c1:
    iaa = st.slider(
        FEATURE_LABELS["IAA"], 0.0, 10.0, 5.0, 0.1,
        help="Avalia o desempenho escolar geral do aluno."
    )
    ieg = st.slider(
        FEATURE_LABELS["IEG"], 0.0, 10.0, 5.0, 0.1,
        help="Mede o nível de participação e envolvimento nas atividades."
    )

with c2:
    ips = st.slider(
        FEATURE_LABELS["IPS"], 0.0, 10.0, 5.0, 0.1,
        help="Representa aspectos emocionais, sociais e de contexto do aluno."
    )
    ida = st.slider(
        FEATURE_LABELS["IDA"], 0.0, 10.0, 5.0, 0.1,
        help="Indica a evolução e o desenvolvimento acadêmico ao longo do tempo."
    )

with c3:
    ipp = st.slider(
        FEATURE_LABELS["IPP"], 0.0, 10.0, 5.0, 0.1,
        help="Relaciona-se à continuidade e permanência do aluno nos estudos."
    )
    ipv = st.slider(
        FEATURE_LABELS["IPV"], 0.0, 10.0, 5.0, 0.1,
        help="Indica o nível de vulnerabilidade social do aluno."
    )

with c4:
    ian = st.slider(
        FEATURE_LABELS["IAN"], 0.0, 10.0, 5.0, 0.1,
        help="Mostra o quanto o aluno precisa de apoio adicional."
    )
    inde = st.slider(
        FEATURE_LABELS["INDE"], 0.0, 10.0, 5.0, 0.1,
        help="Representa o nível geral de desenvolvimento educacional."
    )

if st.button("Prever indicação de bolsa"):
    entrada = pd.DataFrame([{
        "IAA": iaa,
        "IEG": ieg,
        "IPS": ips,
        "IDA": ida,
        "IPP": ipp,
        "IPV": ipv,
        "IAN": ian,
        "INDE": inde,
    }])

    if hasattr(scaler, "feature_names_in_"):
        entrada = entrada[scaler.feature_names_in_]
    else:
        entrada = entrada[FEATURES]

    entrada_scaled = scaler.transform(entrada)

    pred_raw = model.predict(entrada_scaled)
    prob = float(pred_raw.flatten()[0])
    classe = int(prob >= 0.5)

    st.metric("Probabilidade estimada de indicação", f"{prob:.2%}")

    if classe == 1:
        st.success("Perfil com maior probabilidade de indicação para bolsa.")
    else:
        st.error("Perfil com menor probabilidade de indicação para bolsa.")

st.divider()
st.subheader("Histórico de treinamento do modelo")

if {"accuracy", "val_accuracy", "loss", "val_loss"}.issubset(hist.columns):
    col_a, col_b = st.columns(2)

    with col_a:
        fig_acc, ax_acc = plt.subplots(figsize=(8, 4))
        ax_acc.plot(hist["accuracy"], label="Treinamento")
        ax_acc.plot(hist["val_accuracy"], label="Validação")
        ax_acc.set_title("Acurácia")
        ax_acc.set_xlabel("Épocas")
        ax_acc.set_ylabel("Acurácia")
        ax_acc.legend()
        st.pyplot(fig_acc, clear_figure=True)

    with col_b:
        fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
        ax_loss.plot(hist["loss"], label="Treinamento")
        ax_loss.plot(hist["val_loss"], label="Validação")
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("Épocas")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        st.pyplot(fig_loss, clear_figure=True)
else:
    st.warning("O arquivo de histórico não possui as colunas esperadas.")

with st.expander("Ver histórico de treinamento"):
    st.dataframe(hist, use_container_width=True)