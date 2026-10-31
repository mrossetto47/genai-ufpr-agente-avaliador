
"""
Projeto: Agente Avaliador de Sentimentos
Disciplina: Agentes de IA — Pós-graduação em GenAI/UFPR
Professor: Vinícius Fülber Garcia

Código desenvolvido a partir do projeto-base disponibilizado pelo professor:
https://github.com/fulbergarciaufpr/GenAI-LangChain-AgenteAvaliador

O bloco inicial de consulta à API foi adaptado para incluir:
- Tratamento básico de erros HTTP;
- Simplificação da conversão JSON;
- Mensagens de diagnóstico;
- Comentários explicativos.
"""

# ============================================================
# BLOCO 1 — OBTENÇÃO DAS AVALIAÇÕES
# Baseado no código fornecido pelo professor.
# ============================================================

import requests

# Endereço da API que disponibiliza as avaliações de produtos.
# Uma API permite que programas troquem informações por meio
# de interfaces e protocolos definidos.
URL_AVALIACOES = "https://apis.codante.io/api/reviews-api/reviews"

# Realiza uma requisição HTTP do tipo GET.
# O timeout evita que a aplicação espere indefinidamente.
resposta = requests.get(URL_AVALIACOES, timeout=15)

# Verifica o status HTTP da resposta.
# Se ocorrer um erro (por exemplo, 404 ou 500), uma exceção
# será lançada, em vez de prosseguir com dados inválidos.
resposta.raise_for_status()

# Converte a resposta JSON em estruturas de dados Python.
# Neste caso, esperamos um dicionário contendo a chave "data".
dados = resposta.json()

# Inicializa uma lista vazia para armazenar somente os textos
# das avaliações, sem os demais campos retornados pela API.
list_review = []

# Percorre os registros presentes na chave "data".
# Cada registro é um dicionário com informações de uma avaliação.
for review in dados["data"]:

    # Extrai o campo "description" e o adiciona à lista.
    list_review.append(review["description"])

# Exibe a quantidade de avaliações obtidas.
print(f"\nTotal de avaliações: {len(list_review)}")

# Enumera as avaliações para facilitar a inspeção dos dados.
# O parâmetro start=1 faz a numeração começar em 1, e não em 0.
for i, avaliacao in enumerate(list_review, start=1):
    print(f"\nAvaliação {i}: {avaliacao}")


# ============================================================
# BLOCO 2 — CONSTRUÇÃO DO PROMPT COM LANGCHAIN
# Implementação baseada nos conceitos da Aula 02.
# ============================================================

from langchain_core.prompts import ChatPromptTemplate

# O ChatPromptTemplate permite criar modelos reutilizáveis
# de mensagens, com instruções fixas e campos variáveis.
#
# Neste projeto, utilizamos dois tipos de mensagem:
#
# system: define a tarefa e as regras de comportamento da LLM.
# human: contém a avaliação que deverá ser classificada.

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um especialista em análise de sentimentos "
        "de avaliações de produtos. "
        "Sua tarefa é classificar cada avaliação como "
        "POSITIVA ou NEGATIVA. "
        "Considere o sentimento predominante do comentário, "
        "mesmo quando houver elogios e críticas simultaneamente. "
        "Responda exclusivamente com uma das palavras: "
        "POSITIVA ou NEGATIVA."
    ),
    (
        "human",
        "{avaliacao}"
    )
])

# ============================================================
# TESTE DO PROMPT — SEM CHAMAR A LLM
# ============================================================

# Seleciona a quinta avaliação recebida da API.
# O índice 4 corresponde ao quinto elemento da lista Python.
avaliacao_teste = list_review[4]

# Substitui a variável {avaliacao} pelo texto selecionado.
# O método invoke() aqui executa apenas a formatação do prompt.
prompt_formatado = prompt.invoke({
    "avaliacao": avaliacao_teste
})

# Exibe as mensagens produzidas pelo template.
print("\n========== TESTE DO PROMPT ==========")
print(prompt_formatado)


# ============================================================
# BLOCO 3 — CONFIGURAÇÃO DO MODELO DE LINGUAGEM
# Integração com o provedor Groq, conforme a Aula 01.
# ============================================================

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Carrega as variáveis definidas no arquivo .env.
# Isso permite utilizar a chave de API sem escrevê-la
# diretamente no código-fonte.
load_dotenv()

# Recupera a chave de API das variáveis de ambiente.
api_key = os.getenv("GROQ_API_KEY")

# Verifica se a chave foi encontrada.
# Caso contrário, interrompe a execução com uma mensagem clara.
if not api_key:
    raise ValueError(
        "Chave GROQ_API_KEY não encontrada. "
        "Verifique o arquivo .env."
    )

# Instancia o modelo de linguagem.
# A temperatura baixa favorece respostas mais consistentes
# em tarefas de classificação.
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=api_key
)

print("\nModelo de linguagem configurado com sucesso.")
