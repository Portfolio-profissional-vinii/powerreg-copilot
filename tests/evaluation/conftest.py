"""
tests/evaluation/conftest.py

Fixtures compartilhadas entre os módulos de avaliação do RAG híbrido.

ESCOPO:
  - hybrid_retriever  : HybridRetriever (Dense + BM25 + RRF) — escopo de sessão
  - retriever_components : componentes internos expostos para testes isolados
  - orchestrator      : CopilotOrchestrator (requer GROQ_API_KEY ou GOOGLE_API_KEY)
  - corpus            : textos e metadados do corpus carregados do disco
  - reports_dir       : diretório de relatórios

REGRA: Não modifica nenhum arquivo de src/.
       Usa apenas imports públicos da aplicação existente.
"""
import os
import sys
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")


# ---------------------------------------------------------------------------
# Fixture: HybridRetriever (Dense + BM25 + RRF)
# Escopo de sessão para evitar recarregar o modelo de embeddings a cada teste
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def hybrid_retriever():
    """Instância real do HybridRetriever configurado para K=10."""
    try:
        from src.vectorstore.hybrid_search import configurar_buscador
        retriever = configurar_buscador(k_resultados=10, k_candidatos=20)
        return retriever
    except Exception as exc:
        pytest.skip(f"HybridRetriever indisponível: {exc}")


@pytest.fixture(scope="session")
def retriever_components(hybrid_retriever):
    """
    Expõe componentes internos do HybridRetriever para testes isolados
    de Dense, BM25 e Hybrid sem recriar o modelo de embeddings.
    """
    return {
        "retriever": hybrid_retriever,
        "client": hybrid_retriever.client,
        "embeddings": hybrid_retriever.embeddings_model,
        "bm25": hybrid_retriever.bm25,
        "corpus_texts": hybrid_retriever.corpus_texts,
        "corpus_metadados": hybrid_retriever.corpus_metadados,
        "collection_name": hybrid_retriever.collection_name,
    }


# ---------------------------------------------------------------------------
# Fixture: CopilotOrchestrator (RAG Agent + LLM)
# Dependência: GROQ_API_KEY ou GOOGLE_API_KEY, Qdrant disponível
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def orchestrator():
    """Instancia real do CopilotOrchestrator."""
    if not os.getenv("GROQ_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GROQ_API_KEY ou GOOGLE_API_KEY não definida — testes LLM ignorados.")
    try:
        from src.agents.graph import CopilotOrchestrator
        orch = CopilotOrchestrator()
        return orch
    except Exception as exc:
        pytest.skip(f"CopilotOrchestrator indisponível: {exc}")


# ---------------------------------------------------------------------------
# Fixture: Corpus carregado (textos + metadados)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def corpus():
    """
    Retorna (corpus_texts, corpus_metadados) na mesma ordem usada pelo embed.py.
    """
    import json
    pasta = ROOT_DIR / "data" / "processed"
    textos, metadados = [], []
    for arquivo in sorted(pasta.glob("*_chunks.json")):
        with open(arquivo, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        for c in chunks:
            textos.append(c["texto"])
            metadados.append(c["metadados"])
    return textos, metadados


# ---------------------------------------------------------------------------
# Fixture: Reports directory
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def reports_dir():
    """Garante que o diretório de relatórios existe."""
    d = ROOT_DIR / "tests" / "evaluation" / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d
