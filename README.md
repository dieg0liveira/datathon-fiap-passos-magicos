# 📊 Datathon - ONG Passos Mágicos

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 📌 Sobre o Projeto

Projeto de análise de dados e modelagem preditiva desenvolvido para o Datathon da ONG Passos Mágicos.

O objetivo é apoiar a identificação de alunos com maior necessidade de apoio educacional, utilizando modelos de Machine Learning aplicados a indicadores acadêmicos e sociais.

---

## 🚀 Funcionalidades

O sistema possui duas principais análises:

### 🔮 Predição de Ponto de Virada

Estima a probabilidade de um aluno atingir um ponto de virada em sua trajetória educacional, considerando fatores como:

- desempenho acadêmico  
- situação psicossocial  
- continuidade nos estudos  
- vulnerabilidade  
- necessidade de apoio  

---

### 🎓 Predição de Indicação de Bolsa

Estima a probabilidade de um aluno ser indicado para receber bolsa de estudos, com base em:

- desempenho acadêmico  
- engajamento  
- desenvolvimento educacional  
- permanência nos estudos  
- contexto social  

---

## 🧠 Modelos utilizados

- Random Forest (Ponto de Virada)  
- Rede Neural (Keras / TensorFlow) para Indicação de Bolsa  

Os modelos já estão treinados e salvos no repositório (.pkl), não sendo necessário novo treinamento para execução do sistema.

---

## 🖥️ Interface

O projeto possui uma interface interativa construída com Streamlit, permitindo a simulação de cenários a partir da entrada de indicadores.

---

## ⚙️ Configuração do Ambiente

### 🐍 Versão do Python (IMPORTANTE)

O projeto foi desenvolvido utilizando:

**Python 3.11**

Versões mais recentes (como 3.12, 3.13 ou 3.14) podem causar falhas, principalmente na instalação do TensorFlow.

---

## ▶️ Como executar o projeto

### 1. Criar ambiente virtual

```bash
python -m venv venv
```

### 2. Ativar ambiente

**Windows:**

```bash
venv\Scripts\activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Executar aplicação

```bash
streamlit run Inicio.py
```

A aplicação será aberta no navegador.

---

## 📓 Notebooks

Os notebooks utilizados no projeto estão disponíveis em:

```
notebooks/ & model/
```

Eles incluem:

- análise exploratória dos dados  
- tratamento e preparação  
- treinamento dos modelos  

---

## ⚠️ Requisitos adicionais para notebooks

Para executar os notebooks, instale também:

```bash
pip install notebook seaborn plotly missingno imbalanced-learn ipykernel
```

---

## 📂 Estrutura do projeto

```
.
├── Inicio.py
├── pages/
├── notebooks/
├── output/
├── *.pkl
├── requirements.txt
└── README.md
```

---

## 📂 Dados

As bases originais utilizadas no projeto não estão incluídas no repositório devido ao tamanho.

O projeto utiliza apenas:

- dados tratados  
- modelos treinados  
- arquivos necessários para execução da aplicação  

---

## ❗ Observações importantes

- Os modelos já estão treinados  
- Não é necessário executar os notebooks para usar o sistema  
- O app funciona de forma independente  

Caso queiram consultar o material usado para tratar as bases e treinar o modelo, segue o link:  
https://drive.google.com/file/d/1eCrv9xa3WHOgo6l4GNYQ24CVN_Z5To87/view

---

## ⚠️ Problemas comuns

### TensorFlow não instala
- Verifique se está usando Python 3.11  

### Erros ao rodar notebooks
- Instale as dependências adicionais indicadas acima  

---

## ✅ Objetivo do projeto

O projeto busca apoiar a tomada de decisão com base em dados, utilizando modelos preditivos para identificar padrões e auxiliar na priorização de alunos com maior necessidade de acompanhamento.  

As previsões geradas não substituem a análise humana, mas funcionam como suporte à decisão.

---

## 👨‍💻 Autor

Projeto desenvolvido por Diego Oliveira
