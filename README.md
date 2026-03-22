<p align="center">
  <img src="https://img.shields.io/badge/Status-Finalizado-green?style=for-the-badge" alt="Status: Finalizado"/>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/Bibliotecas-Scikit--learn%20|%20TensorFlow%20|%20Keras%20|%20Streamlit-orange?style=for-the-badge" alt="Bibliotecas"/>
  <a href="https://youtu.be/6g1-S-4taPg">
    <img src="https://img.shields.io/badge/🎥%20Assistir%20à%20Apresentação-red?style=for-the-badge&logo=youtube&logoColor=white" alt="Assistir à Apresentação"/>
  </a>
</p>

# 📊 Datathon: ONG Passos Mágicos

Este repositório contém o projeto de **análise de dados e modelagem preditiva** desenvolvido para o Datathon da ONG **Passos Mágicos**.

O foco central é utilizar **Inteligência Artificial** para identificar alunos que necessitam de maior apoio educacional e social.

---

## 🎯 O Objetivo

O projeto visa apoiar a tomada de decisão da ONG por meio de modelos de Machine Learning aplicados a indicadores acadêmicos e sociais.

O sistema atua em duas frentes principais:

- 🔄 **Ponto de Virada**  
  Identificar momentos de evolução significativa na trajetória do aluno.

- 🎓 **Indicação de Bolsa**  
  Estimar a probabilidade de elegibilidade para apoio financeiro.

---

## 📄 Storytelling do Projeto

<a href="POSTECH - Fase 5 - Storytelling.pdf">
    <img src="storytelling_preview.jpg" alt="Storytelling Preview"/>
</a>

<br>

<a href="POSTECH - Fase 5 - Storytelling.pdf">
  <img src="https://img.shields.io/badge/📄%20Baixar%20PDF-blue?style=for-the-badge"/>
</a>

## 🚀 Funcionalidades e Modelos

### 🔮 Predição de Ponto de Virada
- **Modelo**: Random Forest (`.pkl`)
- **Fatores analisados**:
  - Desempenho acadêmico
  - Situação psicossocial
  - Continuidade nos estudos
  - Vulnerabilidade social

### 🎓 Predição de Indicação de Bolsa
- **Modelo**: Rede Neural (Keras / TensorFlow)
- **Fatores analisados**:
  - Engajamento
  - Desenvolvimento educacional
  - Permanência
  - Contexto social

---

## 🖥️ Aplicação Interativa (Streamlit)

A aplicação foi desenvolvida utilizando **Streamlit**, permitindo simular cenários e visualizar predições em tempo real de forma simples e intuitiva.

🔗 **Acesse a aplicação online:**

<a href="https://datathon-fiap-paapps-magicos-gap5ickukarcqyvoge7gpe.streamlit.app/">
  <img src="https://img.shields.io/badge/🚀%20Acessar%20App%20Streamlit-green?style=for-the-badge&logo=streamlit&logoColor=white"/>
</a>

---

### 💡 O que você pode fazer no App

- 🔮 Simular o **Ponto de Virada** de alunos  
- 🎓 Avaliar a **probabilidade de Indicação de Bolsa**  
- 📊 Explorar variáveis educacionais e sociais  
- ⚡ Obter predições em tempo real  

---

> 💡 **Dica:** Para melhor experiência, utilize o app em tela cheia e insira diferentes cenários para explorar o comportamento dos modelos.

## 🚀 Instalação

> 💡 Recomendado: utilize um ambiente virtual (`venv`) para garantir o isolamento das dependências.

## 📓 Dependências para análise e treinamento (opcional)

Para executar os notebooks de análise exploratória e treinamento dos modelos, é necessário instalar dependências adicionais:

```bash
pip install plotly seaborn missingno imbalanced-learn ipykernel ipython nbformat
```

### 1. Clone o repositório

```bash
git clone --branch main https://github.com/Carllux/datathon-fiap-passos-magicos.git
cd datathon-fiap-passos-magicos
```

### 2. Crie o ambiente virtual

```bash
python -m venv .venv
```

### 3. Ative o ambiente virtual

- **Windows**:
  ```bash
  .\.venv\Scripts\activate
  ```

- **Linux/macOS**:
  ```bash
  source .venv/bin/activate
  ```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

### 5. Execução

```bash
streamlit run Inicio.py
```


---

### 🛠️ Estratégia de Desenvolvimento

### 1. Notebooks e Análise

Os notebooks em `/notebooks` cobrem todo o pipeline:

- Análise Exploratória de Dados (EDA)
- Tratamento e limpeza
- Engenharia de atributos
- Treinamento e validação

---

### 2. Modelagem

- Modelo baseado em **Random Forest** para classificação tabular
- Modelo de **Deep Learning** para capturar padrões complexos
- Uso de dados educacionais + sociais
