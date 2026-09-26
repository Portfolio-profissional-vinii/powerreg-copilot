"""
tests/evaluation/retrieval/metrics.py

Métricas de retrieval — implementação determinística em Python puro.

Métricas implementadas:
  - hit_at_k             : 1.0 se ao menos 1 resultado relevante em top-K
  - context_precision_at_k : proporção de posições top-K que são relevantes (Precision@K)
  - context_recall_at_k    : proporção de chunks relevantes recuperados em top-K
  - mrr_at_k             : Mean Reciprocal Rank (1/rank do primeiro relevante)
  - ndcg_at_k            : Normalized Discounted Cumulative Gain

NOTA SOBRE IDENTIFICAÇÃO DE CHUNKS:
  O corpus NÃO possui campo 'chunk_id' nos metadados.
  A identificação de relevância é feita por MÓDULO:
    is_relevant(doc) = (doc.metadata['modulo'] == modulo_esperado)

  Isso significa que avaliamos recall em nível de MÓDULO, não de chunk individual.
  Um hit = ao menos um chunk do módulo correto foi recuperado.

  Quando pagina_esperada está definida, computamos também métricas por (modulo, pagina).
  Essas métricas por página são mais rigorosas e registradas separadamente.

ALINHAMENTO COM RAGAS:
  - context_precision_at_k() é equivalente ao Context Precision do Ragas
    (fração de posições relevantes no top-K, ponderada pela posição)
  - context_recall_at_k()    é equivalente ao Context Recall do Ragas
    (fração dos chunks relevantes totais que foram recuperados)
  - mrr_at_k()               é o Mean Reciprocal Rank (padrão IR)
"""
from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Funções de julgamento de relevância
# ---------------------------------------------------------------------------

def is_relevant_by_module(doc_metadata: dict, modulo_esperado: str) -> bool:
    """Considera relevante qualquer chunk do módulo esperado."""
    return doc_metadata.get("modulo", "") == modulo_esperado


def is_relevant_by_module_and_page(
    doc_metadata: dict,
    modulo_esperado: str,
    pagina_esperada: int,
) -> bool:
    """Considera relevante apenas chunks da página esperada dentro do módulo correto."""
    return (
        doc_metadata.get("modulo", "") == modulo_esperado
        and doc_metadata.get("pagina", -1) == pagina_esperada
    )


# ---------------------------------------------------------------------------
# Métricas individuais por pergunta
# ---------------------------------------------------------------------------

def _ranks_relevantes(docs_metadados: list[dict], modulo_esperado: str) -> list[int]:
    """
    Retorna lista de ranks (1-indexed) dos documentos relevantes.
    Relevância definida por módulo.
    """
    return [
        rank
        for rank, meta in enumerate(docs_metadados, start=1)
        if is_relevant_by_module(meta, modulo_esperado)
    ]


def hit_at_k(docs_metadados: list[dict], modulo_esperado: str, k: int) -> float:
    """
    Hit@K: 1.0 se ao menos um documento relevante está em top-K, 0.0 caso contrário.

    Args:
        docs_metadados: Metadados dos documentos recuperados (na ordem de ranking).
        modulo_esperado: Módulo PRODIST esperado para esta pergunta.
        k: Corte de posição.

    Returns:
        1.0 (hit) ou 0.0 (miss).
    """
    return 1.0 if any(
        is_relevant_by_module(meta, modulo_esperado)
        for meta in docs_metadados[:k]
    ) else 0.0


def context_precision_at_k(
    docs_metadados: list[dict],
    modulo_esperado: str,
    k: int,
) -> float:
    """
    Context Precision@K (CP@K) — alinhado com métrica Ragas.

    Mede a proporção de posições top-K que contêm chunks relevantes,
    ponderada pela posição (chunks relevantes mais altos têm mais peso).

    Fórmula:
        CP@K = (1 / |R@K|) * sum_{i ∈ R@K} Precision@i
        onde R@K = conjunto de ranks em que o chunk é relevante
             Precision@i = (chunks relevantes até posição i) / i

    Quando nenhum chunk relevante for recuperado em top-K, retorna 0.0.

    Args:
        docs_metadados: Metadados dos documentos recuperados (na ordem de ranking).
        modulo_esperado: Módulo PRODIST esperado para esta pergunta.
        k: Corte de posição.

    Returns:
        Context Precision@K ∈ [0.0, 1.0].
    """
    n_relevantes_cumulativo = 0
    soma_precision = 0.0
    ranks_relevantes = []

    for i, meta in enumerate(docs_metadados[:k], start=1):
        if is_relevant_by_module(meta, modulo_esperado):
            n_relevantes_cumulativo += 1
            soma_precision += n_relevantes_cumulativo / i
            ranks_relevantes.append(i)

    if not ranks_relevantes:
        return 0.0

    return soma_precision / len(ranks_relevantes)


def context_recall_at_k(
    docs_metadados: list[dict],
    modulo_esperado: str,
    k: int,
    total_relevantes: int | None = None,
) -> float:
    """
    Context Recall@K (CR@K) — alinhado com métrica Ragas.

    Mede qual proporção do ground truth relevante foi recuperada em top-K.

    Fórmula:
        CR@K = (chunks relevantes recuperados em top-K) / (total de chunks relevantes)

    Quando total_relevantes não é fornecido, usa K como denominador
    (recall upperbound = 1.0 quando todos os k recuperados são relevantes).

    Args:
        docs_metadados: Metadados dos documentos recuperados (na ordem de ranking).
        modulo_esperado: Módulo PRODIST esperado para esta pergunta.
        k: Corte de posição.
        total_relevantes: Total de chunks relevantes no corpus para este módulo.
                          Quando fornecido, produz recall real; caso contrário, recall aproximado.

    Returns:
        Context Recall@K ∈ [0.0, 1.0].
    """
    recuperados_relevantes = sum(
        1 for meta in docs_metadados[:k]
        if is_relevant_by_module(meta, modulo_esperado)
    )
    if total_relevantes is None or total_relevantes == 0:
        # Recall aproximado: denominador = K
        return recuperados_relevantes / max(k, 1)
    return min(recuperados_relevantes / total_relevantes, 1.0)


# Alias de compatibilidade com código anterior
def recall_at_k(
    docs_metadados: list[dict],
    modulo_esperado: str,
    k: int,
    total_relevantes: int | None = None,
) -> float:
    """
    Alias para context_recall_at_k() — mantido para compatibilidade retroativa.

    Prefer context_recall_at_k() em novos usos.
    """
    return context_recall_at_k(docs_metadados, modulo_esperado, k, total_relevantes)


def mrr_at_k(docs_metadados: list[dict], modulo_esperado: str, k: int) -> float:
    """
    MRR (Mean Reciprocal Rank) — contribuição por pergunta.

    Calcula o Reciprocal Rank do primeiro documento relevante dentro de top-K.
    O MRR@K médio sobre múltiplas perguntas é calculado por compute_metrics().

    Fórmula:
        RR = 1 / rank_primeiro_relevante  (se rank ≤ K)
        RR = 0.0                           (se nenhum relevante em top-K)

    Args:
        docs_metadados: Metadados dos documentos recuperados (na ordem de ranking).
        modulo_esperado: Módulo PRODIST esperado para esta pergunta.
        k: Corte de posição.

    Returns:
        Reciprocal Rank ∈ [0.0, 1.0].
    """
    for rank, meta in enumerate(docs_metadados[:k], start=1):
        if is_relevant_by_module(meta, modulo_esperado):
            return 1.0 / rank
    return 0.0


# Alias de compatibilidade com código anterior
def reciprocal_rank(docs_metadados: list[dict], modulo_esperado: str, k: int) -> float:
    """
    Alias para mrr_at_k() — mantido para compatibilidade retroativa.

    Prefer mrr_at_k() em novos usos.
    """
    return mrr_at_k(docs_metadados, modulo_esperado, k)


def ndcg_at_k(docs_metadados: list[dict], modulo_esperado: str, k: int) -> float:
    """
    NDCG@K com relevância binária (1 se módulo correto, 0 caso contrário).

    DCG  = sum( rel_i / log2(i+1) )  para i in 1..K
    IDCG = DCG ideal (todos os relevantes no topo)
    NDCG = DCG / IDCG

    Args:
        docs_metadados: Metadados dos documentos recuperados (na ordem de ranking).
        modulo_esperado: Módulo PRODIST esperado para esta pergunta.
        k: Corte de posição.

    Returns:
        NDCG@K ∈ [0.0, 1.0].
    """
    relevances = [
        1 if is_relevant_by_module(meta, modulo_esperado) else 0
        for meta in docs_metadados[:k]
    ]

    dcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(relevances, start=1)
    )

    ideal = sorted(relevances, reverse=True)
    idcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(ideal, start=1)
    )

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Agregação sobre múltiplas perguntas
# ---------------------------------------------------------------------------

def compute_metrics(
    resultados: list[dict[str, Any]],
    k: int,
    corpus_por_modulo: dict[str, int] | None = None,
) -> dict[str, float]:
    """
    Agrega métricas sobre uma lista de resultados de retrieval.

    Args:
        resultados: Lista de dicts com chaves:
            - 'docs_metadados': list[dict] — metadados dos documentos recuperados (na ordem)
            - 'modulo_esperado': str
        k: Valor de K para corte.
        corpus_por_modulo: dict {modulo: n_chunks} para cálculo de context_recall real.
                           Quando None, usa recall aproximado (denominador=K).

    Returns:
        Dict com métricas agregadas, incluindo todas as métricas obrigatórias:
            hit_rate@K, context_precision@K, context_recall@K, mrr@K, ndcg@K, recall@K, n_queries
    """
    hits, cp_scores, cr_scores, mrr_scores, ndcg_scores = [], [], [], [], []

    for r in resultados:
        docs = r["docs_metadados"]
        mod = r["modulo_esperado"]

        total_rel = corpus_por_modulo.get(mod) if corpus_por_modulo else None

        hits.append(hit_at_k(docs, mod, k))
        cp_scores.append(context_precision_at_k(docs, mod, k))
        cr_scores.append(context_recall_at_k(docs, mod, k, total_rel))
        mrr_scores.append(mrr_at_k(docs, mod, k))
        ndcg_scores.append(ndcg_at_k(docs, mod, k))

    n = len(resultados) or 1
    return {
        f"hit_rate@{k}": sum(hits) / n,
        f"context_precision@{k}": sum(cp_scores) / n,
        f"context_recall@{k}": sum(cr_scores) / n,
        f"mrr@{k}": sum(mrr_scores) / n,
        f"ndcg@{k}": sum(ndcg_scores) / n,
        # Alias de compatibilidade com relatórios anteriores
        f"recall@{k}": sum(cr_scores) / n,
        "n_queries": n,
    }


def compute_all_k(
    resultados: list[dict[str, Any]],
    ks: list[int] = [1, 3, 5, 10, 20],
    corpus_por_modulo: dict[str, int] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Calcula métricas para múltiplos valores de K.

    Args:
        resultados: Lista de resultados (mesmo formato de compute_metrics).
        ks: Lista de valores K a avaliar.
        corpus_por_modulo: Opcional — {modulo: n_chunks} para recall real.

    Returns:
        Dict {str(k): metricas_dict}.
    """
    return {str(k): compute_metrics(resultados, k, corpus_por_modulo) for k in ks}


def formatar_tabela_metricas(resultados_por_retriever: dict[str, dict]) -> str:
    """
    Formata tabela markdown de comparação entre retrievers.

    Args:
        resultados_por_retriever: {nome_retriever: {k: {metrica: valor}}}

    Returns:
        String markdown com tabela comparativa.
    """
    linhas = [
        "| Retriever | K | Hit@K | CP@K | CR@K | MRR@K | NDCG@K |",
        "|-----------|---|------:|-----:|-----:|------:|-------:|",
    ]
    for retriever, por_k in resultados_por_retriever.items():
        for k, metricas in por_k.items():
            hit = metricas.get(f"hit_rate@{k}", 0)
            cp  = metricas.get(f"context_precision@{k}", 0)
            cr  = metricas.get(f"context_recall@{k}", 0)
            mrr = metricas.get(f"mrr@{k}", 0)
            ndcg = metricas.get(f"ndcg@{k}", 0)
            linhas.append(
                f"| {retriever} | {k} | {hit:.3f} | {cp:.3f} | {cr:.3f} | {mrr:.3f} | {ndcg:.3f} |"
            )
    return "\n".join(linhas)
