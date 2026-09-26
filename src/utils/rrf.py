"""
src/utils/rrf.py

Módulo de Reciprocal Rank Fusion (RRF) para consolidação e deduplicação de rankings.

Fundamento Teórico:
    O Reciprocal Rank Fusion (Cormack et al., 2009) é um método não-supervisionado
    de fusão de rankings que combina múltiplos resultados de busca (ex: sub-queries,
    fontes densas e léxicas) sem depender de calibração ou normalização prévia de scores.

    Fórmula:
        RRF_Score(d) = sum( 1 / (k + rank_i(d)) )

    Onde:
        - d: documento avaliado
        - rank_i(d): posição do documento d no ranking da i-ésima query (1-indexed)
        - k: constante de amortecimento (padrão da literatura = 60)
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Valor padrão da literatura (Cormack, Clarke & Buettcher, SIGIR 2009)
DEFAULT_RRF_K: int = 60


def calcular_score_rrf(rank: int, k: int = DEFAULT_RRF_K) -> float:
    """
    Calcula a contribuição de RRF para uma determinada posição no ranking.

    Args:
        rank: Posição do item no ranking (1-indexed: 1 para o primeiro colocado).
        k: Constante de suavização RRF (padrão: 60).

    Returns:
        Score RRF correspondente: 1.0 / (k + rank).
    """
    if rank < 1:
        raise ValueError(f"O rank deve ser >= 1 (1-indexed), recebido: {rank}")
    return 1.0 / (k + rank)


def _obter_chave_deduplicacao(doc: Document, id_key: str = "chunk_id") -> Union[str, int]:
    """
    Extrai ou calcula uma chave única determinística para deduplicação do documento.

    Prioridade:
        1. metadata[id_key] (ex: chunk_id ou id de ponto Qdrant)
        2. metadata['id'] ou metadata['document_id']
        3. Composição de módulo, página e hash do texto
        4. Hash SHA256 do conteúdo da página
    """
    meta = doc.metadata or {}
    
    # 1. Chave explícita configurada
    if id_key in meta and meta[id_key] is not None:
        return f"id:{meta[id_key]}"
    
    # 2. Alternativas comuns de chave primária
    for alt_key in ("id", "doc_id", "document_id"):
        if alt_key in meta and meta[alt_key] is not None:
            return f"{alt_key}:{meta[alt_key]}"
            
    # 3. Composição de módulo + página + texto
    modulo = meta.get("modulo", "")
    pagina = meta.get("pagina", "")
    texto_snippet = doc.page_content.strip()
    return f"chunk:{modulo}:{pagina}:{hash(texto_snippet)}"


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = DEFAULT_RRF_K,
    top_n: int = 10,
    id_key: str = "chunk_id",
) -> List[Document]:
    """
    Aplica Reciprocal Rank Fusion (RRF) sobre múltiplas listas ranqueadas de Documentos.

    Consolida documentos recuperados por diferentes sub-queries ou mecanismos de busca,
    soma as pontuações RRF das aparições de cada documento, remove duplicatas e retorna
    os top_n documentos ordenados pelo score decrescente.

    Args:
        ranked_lists: Lista contendo as listas de Document retornadas por cada query/fonte.
        k: Parâmetro de suavização RRF (padrão: 60). Valores maiores reduzem a vantagem
           de posições de topo sobre posições intermediárias.
        top_n: Quantidade máxima de documentos consolidados a retornar (padrão: 10).
        id_key: Nome do campo nos metadados que identifica unicamente o chunk (padrão: 'chunk_id').

    Returns:
        Lista com até top_n Documentos deduplicados, ordenados por score RRF decrescente,
        com os metadados enriquecidos:
            - 'rrf_score': pontuação RRF agregada (float)
            - 'rrf_occurrences': quantidade de listas nas quais o documento apareceu (int)
            - 'rrf_ranks': lista dos ranks (1-indexed) ocupados em cada lista (list[int])
    """
    if not ranked_lists:
        return []

    # Mapas para acumulação
    scores: Dict[Union[str, int], float] = {}
    docs_map: Dict[Union[str, int], Document] = {}
    ranks_map: Dict[Union[str, int], List[int]] = {}

    for list_idx, doc_list in enumerate(ranked_lists):
        if not doc_list:
            continue

        # Evita contabilizar duplicatas dentro da MESMA lista mais de uma vez
        vistos_nesta_lista = set()

        for rank, doc in enumerate(doc_list, start=1):
            doc_key = _obter_chave_deduplicacao(doc, id_key=id_key)
            if doc_key in vistos_nesta_lista:
                continue
            vistos_nesta_lista.add(doc_key)

            parcela_rrf = calcular_score_rrf(rank, k=k)

            # Acumula o score RRF
            scores[doc_key] = scores.get(doc_key, 0.0) + parcela_rrf

            # Registra o rank ocupado nesta lista
            if doc_key not in ranks_map:
                ranks_map[doc_key] = []
            ranks_map[doc_key].append(rank)

            # Guarda o documento com seus metadados originais (faz cópia para não mutar entrada)
            if doc_key not in docs_map:
                docs_map[doc_key] = copy.deepcopy(doc)

    if not scores:
        return []

    # Ordena chaves por score RRF decrescente
    chaves_ordenadas = sorted(scores.keys(), key=lambda chave: scores[chave], reverse=True)

    # Constrói a lista final enriquecida
    resultados: List[Document] = []
    for doc_key in chaves_ordenadas[:top_n]:
        doc = docs_map[doc_key]
        if doc.metadata is None:
            doc.metadata = {}

        score_final = scores[doc_key]
        ranks_atingidos = ranks_map[doc_key]

        doc.metadata["rrf_score"] = float(score_final)
        doc.metadata["rrf_occurrences"] = len(ranks_atingidos)
        doc.metadata["rrf_ranks"] = ranks_atingidos

        resultados.append(doc)

    logger.debug(
        "RRF consolidou %d listas (%d docs únicos) em top-%d resultados.",
        len(ranked_lists),
        len(scores),
        len(resultados),
    )

    return resultados
