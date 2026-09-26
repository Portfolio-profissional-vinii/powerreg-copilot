"""
Script de integração para validar o pipeline completo de reranking.
Conecta ao Qdrant (via QDRANT_URL=http://localhost:6333) e executa
o fluxo: Query -> Híbrido (Dense + BM25) -> RRF -> Cross-Encoder -> Top-K Final.

Uso:
    .venv\\Scripts\\python.exe tests\\integration_reranker.py
"""
import os
import sys
import time
from pathlib import Path

# Garante que o root do projeto está no path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Carrega o .env
from dotenv import load_dotenv
load_dotenv()

from src.vectorstore.hybrid_search import configurar_buscador

QUERIES = [
    "Quais são os limites de tensão aceitáveis para a rede de distribuição?",
    "O agente de distribuição é obrigado a conectar consumidores de baixa tensão?",
    "Qual o prazo para reestabelecimento do fornecimento após interrupção?",
]

def linha(char="-", n=72):
    print(char * n)

def testar_com_reranker(query: str, retriever_sem, retriever_com):
    print(f"\nQUERY: {query}")
    linha()

    # --- Resultado SEM reranker (apenas RRF) ---
    docs_sem = retriever_sem.invoke(query)
    print("\n[SEM Reranker - apenas RRF]")
    for i, doc in enumerate(docs_sem, 1):
        meta = doc.metadata
        snippet = doc.page_content[:120].replace("\n", " ")
        print(f"  #{i} [{meta.get('modulo', '?')} | Pag.{meta.get('pagina', '?')}]: {snippet}...")

    # --- Resultado COM reranker (RRF + Cross-Encoder) ---
    docs_com = retriever_com.invoke(query)
    print("\n[COM Reranker - RRF + Cross-Encoder]")
    for i, doc in enumerate(docs_com, 1):
        meta = doc.metadata
        snippet = doc.page_content[:120].replace("\n", " ")
        print(f"  #{i} [{meta.get('modulo', '?')} | Pag.{meta.get('pagina', '?')}]: {snippet}...")

    # --- Detecção de mudança de ranking ---
    ids_sem = [d.metadata.get("chunk_id") for d in docs_sem]
    ids_com = [d.metadata.get("chunk_id") for d in docs_com]
    if ids_sem != ids_com:
        print("\n  [OK] Reranker ALTEROU a ordem dos resultados.")
    else:
        print("\n  [--] Reranker manteve a mesma ordem do RRF para esta query.")

    linha("=")


if __name__ == "__main__":
    print("=" * 72)
    print("  TESTE DE INTEGRACAO - RERANKING COM CROSS-ENCODER")
    print("=" * 72)

    print("\n[1/2] Inicializando retriever SEM reranker...")
    t0 = time.time()
    retriever_sem = configurar_buscador(k_resultados=5, k_candidatos=20, k_rrf=20, use_reranker=False)
    print(f"      Pronto em {time.time() - t0:.1f}s")

    print("\n[2/2] Inicializando retriever COM reranker (Cross-Encoder local)...")
    t0 = time.time()
    retriever_com = configurar_buscador(k_resultados=5, k_candidatos=20, k_rrf=20, use_reranker=True)
    print(f"      Pronto em {time.time() - t0:.1f}s")

    for query in QUERIES:
        t_query = time.time()
        testar_com_reranker(query, retriever_sem, retriever_com)
        print(f"  Tempo total da query: {time.time() - t_query:.2f}s")

    print("\nTeste de integracao concluido.\n")
