"""
src/retrieval/multi_query.py

Módulo de Multi-Query Retrieval integrado para o Utilities Copilot (PRODIST/ANEEL).

Arquitetura:
                QUERY ORIGINAL DO USUÁRIO
                           │
                           ▼
                 [QueryExpander Agent]
                 (Tradução Normativa,
                  Módulo PRODIST,
                  Prazos/Penalidades)
                           │
             ┌─────────────┼─────────────┬─────────────┐
             ▼             ▼             ▼             ▼
          Query 0       Query 1       Query 2       Query 3
         (Original)   (Normativa)    (Módulo)      (Prazos)
             │             │             │             │
             └─────────────┼─────────────┴─────────────┘
                           │
                           ▼ (ThreadPoolExecutor / asyncio.gather)
             [Execução Concorrente de Busca Híbrida]
               Dense (Qdrant) + Lexical (BM25) por sub-query
                           │
                           ▼
          [Reciprocal Rank Fusion (RRF) & Deduplicação]
                 Fórmula: sum( 1 / (k + rank_i) )
                 Preservação de metadados e chunk_id
                           │
                           ▼
                 Top-N Chunks Finais
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from src.agents.query_expander import QueryExpander
from src.utils.rrf import reciprocal_rank_fusion, DEFAULT_RRF_K
from src.vectorstore.hybrid_search import HybridRetriever, configurar_buscador

logger = logging.getLogger(__name__)


class MultiQueryHybridRetriever(BaseRetriever):
    """
    Retriever avançado Multi-Query com execução concorrente e fusão RRF.

    Combina:
      1. Expansão semântica via QueryExpander especializado em normas da ANEEL/PRODIST;
      2. Disparo concorrente de buscas híbridas (Qdrant + BM25) para cada query gerada;
      3. Fusão e deduplicação via Reciprocal Rank Fusion (RRF) com parâmetro k configurável.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hybrid_retriever: Any = Field(
        description="Instância do HybridRetriever (Dense + BM25) para execução das buscas"
    )
    query_expander: Any = Field(
        description="Agente expansor de queries regulatórias"
    )
    k: int = Field(
        default=10,
        description="Número final de documentos consolidados a retornar após fusão RRF",
    )
    rrf_k: int = Field(
        default=DEFAULT_RRF_K,
        description="Parâmetro k de amortecimento da fórmula de RRF (padrão: 60)",
    )
    max_workers: int = Field(
        default=4,
        description="Número máximo de threads para execução concorrente de queries",
    )
    incluir_original: bool = Field(
        default=True,
        description="Se deve incluir a query original do usuário junto com as sub-queries geradas",
    )

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Execução síncrona com paralelismo via ThreadPoolExecutor.
        """
        # 1. Gera sub-queries especializadas (com fallback automático interno)
        queries = self.query_expander.obter_todas_as_queries(
            query, incluir_original=self.incluir_original
        )
        logger.info(
            "MultiQueryRetriever: Pergunta original expandida em %d consultas: %s",
            len(queries),
            queries,
        )

        if not queries:
            return []

        if len(queries) == 1:
            docs = self.hybrid_retriever.invoke(queries[0])
            return reciprocal_rank_fusion(
                ranked_lists=[docs],
                k=self.rrf_k,
                top_n=self.k,
                id_key="chunk_id",
            )

        # 2. Execução concorrente das buscas híbridas para cada query
        listas_de_docs: List[List[Document]] = []
        num_workers = min(self.max_workers, len(queries))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_query = {
                executor.submit(self.hybrid_retriever.invoke, q): q for q in queries
            }

            for future in as_completed(future_to_query):
                q_atual = future_to_query[future]
                try:
                    docs = future.result()
                    listas_de_docs.append(docs)
                    logger.debug("Busca concluída para query '%s': %d docs recuperados.", q_atual, len(docs))
                except Exception as exc:
                    logger.error(
                        "Erro durante busca concorrente da query '%s': %s",
                        q_atual,
                        exc,
                    )

        # 3. Fusão e deduplicação de todos os rankings via RRF
        docs_consolidados = reciprocal_rank_fusion(
            ranked_lists=listas_de_docs,
            k=self.rrf_k,
            top_n=self.k,
            id_key="chunk_id",
        )

        return docs_consolidados

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Execução assíncrona não-bloqueante usando asyncio.gather.
        """
        loop = asyncio.get_running_loop()

        # 1. Expansão da query
        queries = await loop.run_in_executor(
            None,
            self.query_expander.obter_todas_as_queries,
            query,
            self.incluir_original,
        )

        if not queries:
            return []

        if len(queries) == 1:
            docs = await loop.run_in_executor(None, self.hybrid_retriever.invoke, queries[0])
            return reciprocal_rank_fusion(
                ranked_lists=[docs],
                k=self.rrf_k,
                top_n=self.k,
                id_key="chunk_id",
            )

        # 2. Execução concorrente assíncrona via asyncio.gather
        tarefas = [
            loop.run_in_executor(None, self.hybrid_retriever.invoke, q)
            for q in queries
        ]
        resultados = await asyncio.gather(*tarefas, return_exceptions=True)

        listas_validas: List[List[Document]] = []
        for q, res in zip(queries, resultados):
            if isinstance(res, Exception):
                logger.error("Erro na busca assíncrona para query '%s': %s", q, res)
            elif isinstance(res, list):
                listas_validas.append(res)

        # 3. Fusão RRF
        return reciprocal_rank_fusion(
            ranked_lists=listas_validas,
            k=self.rrf_k,
            top_n=self.k,
            id_key="chunk_id",
        )


def configurar_multi_query_retriever(
    top_n: int = 10,
    k_candidatos_base: int = 15,
    rrf_k: int = DEFAULT_RRF_K,
    max_workers: int = 4,
    incluir_original: bool = True,
    expander: Optional[QueryExpander] = None,
    hybrid_retriever: Optional[HybridRetriever] = None,
) -> MultiQueryHybridRetriever:
    """
    Fábrica para instanciar o MultiQueryHybridRetriever configurado.

    Args:
        top_n: Quantidade final de documentos a retornar após consolidação.
        k_candidatos_base: Quantidade de candidatos retornados por cada sub-query antes do RRF.
        rrf_k: Constante k do RRF (padrão 60).
        max_workers: Threads concorrentes para paralelismo das queries.
        incluir_original: Se a query original deve ser incluída no pool de buscas.
        expander: Instância opcional de QueryExpander pré-configurada.
        hybrid_retriever: Instância opcional de HybridRetriever pré-configurada.

    Returns:
        Instância pronta de MultiQueryHybridRetriever.
    """
    if hybrid_retriever is None:
        # Cada sub-query traz k_candidatos_base para enriquecer o pool de fusão RRF
        hybrid_retriever = configurar_buscador(
            k_resultados=k_candidatos_base,
            k_candidatos=20,
        )

    if expander is None:
        expander = QueryExpander()

    return MultiQueryHybridRetriever(
        hybrid_retriever=hybrid_retriever,
        query_expander=expander,
        k=top_n,
        rrf_k=rrf_k,
        max_workers=max_workers,
        incluir_original=incluir_original,
    )


if __name__ == "__main__":
    print("Inicializando Multi-Query Hybrid Retriever...")
    retriever = configurar_multi_query_retriever(top_n=5)
    pergunta_demo = "Qual o prazo para análise e ressarcimento de aparelhos queimados?"
    print(f"\nPergunta: {pergunta_demo}\nBuscando concorrentemente...")
    
    docs = retriever.invoke(pergunta_demo)
    print(f"\nTop-{len(docs)} Documentos Consolidados via RRF:")
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        print(f"#{i} [Score RRF: {m.get('rrf_score', 0.0):.5f} | Ocorrências: {m.get('rrf_occurrences', 1)}]")
        print(f"    Módulo: {m.get('modulo', '?')} | Página: {m.get('pagina', '?')} | Chunk ID: {m.get('chunk_id', '?')}")
        trecho = doc.page_content.replace('\n', ' ')[:140]
        try:
            print(f"    Texto: {trecho}...\n")
        except UnicodeEncodeError:
            print("    Texto: [caracteres especiais omitidos]\n")
