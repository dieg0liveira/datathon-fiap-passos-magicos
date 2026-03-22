import streamlit as st

st.set_page_config(
    page_title="Início | Datathon ONG Passos Mágicos",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Datathon - ONG Passos Mágicos")

st.markdown("""
Este sistema utiliza modelos de Machine Learning para apoiar a análise educacional de alunos da ONG Passos Mágicos.

A proposta é transformar indicadores educacionais e sociais em previsões que ajudem na identificação de alunos com maior necessidade de apoio, potencial de desenvolvimento e probabilidade de indicação para bolsa.
""")

st.divider()

st.subheader("📌 Como utilizar o sistema")

st.info("""
Preencha os indicadores em cada página utilizando a escala de 0 a 10:

- **0 a 3 → Baixo**
- **4 a 7 → Médio**
- **8 a 10 → Alto**
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔮 Ponto de Virada")
    st.markdown("""
Nesta página, você pode estimar a probabilidade de um aluno atingir um **ponto de virada** em sua trajetória educacional.

A análise considera fatores como:
- desempenho acadêmico
- situação psicossocial
- continuidade nos estudos
- vulnerabilidade
- necessidade de apoio
""")
    if st.button("Acessar página de Ponto de Virada"):
        st.switch_page("pages/1 Ponto de virada.py")

with col2:
    st.markdown("### 🎓 Indicação de Bolsa")
    st.markdown("""
Nesta página, você pode estimar a probabilidade de um aluno ser **indicado para receber bolsa de estudos**.

O modelo considera fatores como:
- desempenho acadêmico
- engajamento
- desenvolvimento educacional
- permanência nos estudos
- contexto social
""")
    if st.button("Acessar página de Indicação de Bolsa"):
        st.switch_page("pages/2 Indicacão de bolsa.py")

st.divider()

st.subheader("✅ Objetivo do projeto")

st.markdown("""
O objetivo deste projeto é apoiar a tomada de decisão com base em dados, utilizando modelos preditivos para complementar a análise de perfis educacionais.

As previsões geradas não substituem a avaliação humana, mas servem como apoio para identificar padrões e priorizar atenção a determinados perfis de alunos.
""")

st.success("Você também pode navegar pelas páginas usando o menu lateral.")
