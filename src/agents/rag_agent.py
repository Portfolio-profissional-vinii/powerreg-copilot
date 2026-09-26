"""
src/agents/rag_agent.py

Agente Regulatório — RAG sobre normativas ANEEL/PRODIST.

LLM: Groq openai/gpt-oss-20b (via _build_llm do graph.py)
Retriever: HybridRetriever (Dense Qdrant + BM25 + RRF)

Prompt de sistema reforçado com regras anti-alucinação para garantir
grounding estrito nas normativas recuperadas (sem invenção de artigos,
módulos ou páginas que não estejam no contexto).
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from src.vectorstore.hybrid_search import configurar_buscador
from src.retrieval.multi_query import configurar_multi_query_retriever
from src.agents.llm_factory import _build_llm


# ---------------------------------------------------------------------------
# System prompt — grounding estrito PRODIST (anti-alucinação)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Você é o Copilot Regulatório da Total Utiliti, especialista em normas da ANEEL e PRODIST.

Responda à dúvida do usuário EXCLUSIVAMENTE com base no contexto de normativas fornecido abaixo.

REGRAS OBRIGATÓRIAS (violação = resposta inválida):
1. Não invente informações, valores, artigos, módulos ou páginas que não estejam no contexto.
2. Cada informação apresentada deve ser fundamentada em uma fonte do contexto.
3. Cite a fonte no formato [Módulo X, Página Y] após cada afirmação relevante.
4. Use exatamente o módulo e a página informados na fonte correspondente.
5. Se a página não estiver disponível no contexto, escreva [Página não identificada].
6. Nunca escreva 'Página N/A' quando a página estiver disponível no contexto.
7. Não crie links, HTML, SVG ou referências como [svg](...).
8. Se a informação solicitada não estiver no contexto, responda: "Não encontrei essa informação nas normativas fornecidas."

Contexto das Normativas:
{context}"""


def formatar_documentos(docs: list) -> str:
    """
    Formata documentos recuperados com metadados para o LLM.

    Args:
        docs: Lista de Document do LangChain com metadados de módulo e página.

    Returns:
        String formatada com cada chunk precedido de sua fonte.
    """
    textos = []
    for doc in docs:
        modulo = doc.metadata.get("modulo", "Módulo Desconhecido")
        pagina = doc.metadata.get("pagina", "N/A")
        textos.append(f"[Fonte: {modulo}, Página: {pagina}]\n{doc.page_content}")
    return "\n\n".join(textos)


def iniciar_copilot(use_multi_query: bool = True, k_resultados: int = 8):
    """
    Inicializa e retorna a cadeia RAG regulatória.

    Componentes:
        - MultiQueryHybridRetriever (Dense + BM25 + RRF + Multi-Query Concorrente)
        - ChatGroq openai/gpt-oss-20b (ou Gemini 2.0 Flash como fallback)
        - Prompt com grounding estrito PRODIST

    Args:
        use_multi_query: Se True, ativa a expansão em sub-queries e busca concorrente.
        k_resultados: Quantidade de documentos consolidados no contexto final (padrão 8).

    Returns:
        Cadeia LangChain (Runnable) que aceita uma pergunta (str) e retorna resposta (str).
    """
    llm = _build_llm(temperature=0.1)

    if use_multi_query:
        retriever = configurar_multi_query_retriever(top_n=k_resultados)
    else:
        retriever = configurar_buscador(k_resultados=k_resultados)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("user", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | formatar_documentos,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    print("Ligando o motor de busca híbrida (Dense + BM25 com RRF)...")
    try:
        agente = iniciar_copilot()
        print("\nCopilot ANEEL iniciado! Digite 'sair' para encerrar.")
        try:
            pergunta = input("\nVocê: ")
        except EOFError:
            sys.exit(0)

        if pergunta.strip().lower() in ["sair", "exit", "quit"] or not pergunta.strip():
            sys.exit(0)

        print("Pesquisando nas leis e formulando a resposta...\n")
        resposta = agente.invoke(pergunta)

        print("Copilot Total Utiliti:")
        print(resposta)

    except Exception as e:
        print(f"Erro ao inicializar o assistente: {e}")