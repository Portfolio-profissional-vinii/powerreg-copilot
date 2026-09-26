"""
src/retrieval — Módulo de recuperação de documentos (Dense, BM25, Multi-Query).
"""
from src.retrieval.multi_query import MultiQueryHybridRetriever, configurar_multi_query_retriever

__all__ = ["MultiQueryHybridRetriever", "configurar_multi_query_retriever"]
