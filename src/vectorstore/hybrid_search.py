"""
Hybrid Search — Dense (Qdrant) + Lexical (rank_bm25) com RRF manual em Python.

Arquitetura:
        QUERY
          |
  +-------+-------+
  |               |
  v               v
Dense Search   BM25 Search
  Qdrant        rank_bm25
  (top-N)       (top-N)
  |               |
  +-------+-------+
          |
          v
     RRF manual
     Python puro
          |
          v
       Top-K
"""
import json
import pickle
import re
from pathlib import Path
from typing import Any, List
import os
import torch
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import ConfigDict, Field
from qdrant_client import QdrantClient

# Parâmetro RRF: controla suavização do ranking.
# 60 é o valor padrão da literatura (Cormack et al., 2009).
RRF_K: int = 60

_qdrant_client_singleton = None


def tokenizar_pt(texto: str) -> list[str]:
    """
    Tokenização básica para português.
    DEVE ser idêntica à usada em embed.py para garantir
    que o índice BM25 e as queries usem o mesmo vocabulário.
    """
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto)
    return [t for t in texto.split() if len(t) > 2]


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    """Calcula o score RRF para um dado rank (1-indexed)."""
    return 1.0 / (k + rank)


class HybridRetriever(BaseRetriever):
    """
    Retriever híbrido: Dense Search (Qdrant) + BM25 (rank_bm25) + RRF manual.

    O RRF é calculado em Python puro, combinando rankings de duas fontes
    independentes. Não usa SparseVector, FusionQuery ou hash de tokens.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any = Field(description="Instância do QdrantClient")
    embeddings_model: Any = Field(description="Modelo HuggingFace de embeddings")
    bm25: Any = Field(description="Modelo BM25Okapi carregado do pickle")
    corpus_texts: List[str] = Field(description="Lista de textos do corpus (mesma ordem do BM25)")
    corpus_metadados: List[dict] = Field(description="Lista de metadados do corpus")
    collection_name: str = "prodist_normativas"
    k: int = 5
    k_candidatos: int = 20
    k_rrf: int = 20
    cross_encoder: Any = Field(default=None, description="Modelo CrossEncoder para reranking")

    def _busca_dense(self, query: str) -> list[tuple[int, float]]:
        """
        Busca densa via Qdrant.
        Retorna lista de (chunk_id, score) ordenada por score decrescente.
        chunk_id corresponde ao ID do ponto no Qdrant, que é igual ao índice
        do chunk na lista do corpus (garantido pelo embed.py).
        """
        dense_vec = self.embeddings_model.embed_query(query)
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vec,
            limit=self.k_candidatos,
            with_payload=False,  # Não precisamos do payload aqui, só do ID e score
        )
        return [(int(p.id), p.score) for p in res.points]

    def _busca_bm25(self, query: str) -> list[tuple[int, float]]:
        """
        Busca lexical via BM25.
        Retorna lista de (chunk_id, score) ordenada por score decrescente,
        limitada a k_candidatos resultados com score > 0.
        """
        tokens = tokenizar_pt(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # Enumerate para obter o índice original (= chunk_id no Qdrant)
        indexed_scores = [(i, float(s)) for i, s in enumerate(scores)]
        # Filtrar score > 0 e ordenar por score decrescente
        positivos = [(i, s) for i, s in indexed_scores if s > 0.0]
        positivos.sort(key=lambda x: x[1], reverse=True)
        return positivos[: self.k_candidatos]

    def _aplicar_rrf(
        self,
        dense_results: list[tuple[int, float]],
        bm25_results: list[tuple[int, float]],
    ) -> list[int]:
        """
        Aplica Reciprocal Rank Fusion (RRF) combinando dois rankings.
        RRF_score(d) = sum( 1 / (RRF_K + rank_i(d)) )

        Retorna lista de chunk_ids ordenada por score RRF decrescente,
        limitada a self.k_rrf resultados (ou self.k se não usar cross-encoder).
        """
        rrf_scores: dict[int, float] = {}

        # Contribuição do ranking Dense (rank é 1-indexed)
        for rank, (chunk_id, _score) in enumerate(dense_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + _rrf_score(rank)

        # Contribuição do ranking BM25 (rank é 1-indexed)
        for rank, (chunk_id, _score) in enumerate(bm25_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + _rrf_score(rank)

        # Ordenar por score RRF decrescente e retornar top-k IDs
        ordenados = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        limite = self.k_rrf if self.cross_encoder else self.k
        return [chunk_id for chunk_id, _ in ordenados[: limite]]

    def _aplicar_cross_encoder(self, query: str, top_ids: list[int]) -> list[int]:
        """
        Aplica reranking usando um CrossEncoder na lista de IDs filtrada pelo RRF.
        Retorna a lista reordenada cortada em self.k.
        """
        if not self.cross_encoder or not top_ids:
            return top_ids[: self.k]
            
        pairs = []
        valid_ids = []
        for chunk_id in top_ids:
            if 0 <= chunk_id < len(self.corpus_texts):
                pairs.append([query, self.corpus_texts[chunk_id]])
                valid_ids.append(chunk_id)
                
        if not pairs:
            return []
            
        scores = self.cross_encoder.predict(pairs)
        
        # Junta o id com o novo score do reranker
        id_score_pairs = list(zip(valid_ids, scores))
        # Reordena por score decrescente do reranker
        id_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk_id for chunk_id, _ in id_score_pairs[: self.k]]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        dense_results = self._busca_dense(query)
        bm25_results = self._busca_bm25(query)

        # 1. Obter Top-N do RRF
        top_ids_rrf = self._aplicar_rrf(dense_results, bm25_results)
        
        # 2. Reordenar com Cross-Encoder para extrair Top-K
        top_ids_final = self._aplicar_cross_encoder(query, top_ids_rrf)

        docs = []
        for chunk_id in top_ids_final:
            # chunk_id é o índice do chunk no corpus (garantido pelo embed.py id=i)
            if 0 <= chunk_id < len(self.corpus_texts):
                meta = dict(self.corpus_metadados[chunk_id])
                meta["chunk_id"] = chunk_id
                docs.append(
                    Document(
                        page_content=self.corpus_texts[chunk_id],
                        metadata=meta,
                    )
                )
        return docs


def _carregar_corpus(pasta_processados: Path) -> tuple[list[str], list[dict]]:
    """
    Carrega textos e metadados do corpus na MESMA ORDEM usada pelo embed.py.
    sorted() sobre os arquivos garante ordem determinística e reproduzível.
    """
    textos = []
    metadados = []
    for arquivo in sorted(pasta_processados.glob("*_chunks.json")):
        with open(arquivo, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        for c in chunks:
            textos.append(c["texto"])
            metadados.append(c["metadados"])
    return textos, metadados


def configurar_buscador(
    k_resultados: int = 5,
    k_candidatos: int = 20,
    k_rrf: int = 20,
    use_reranker: bool = True
) -> HybridRetriever:
    """
    Inicializa e retorna o HybridRetriever.

    Args:
        k_resultados: número de documentos finais a retornar (após CrossEncoder/RRF).
        k_candidatos: número de candidatos por fonte (Dense e BM25) antes do RRF.
        k_rrf: número de candidatos passados do RRF para o CrossEncoder.
        use_reranker: Se True, utiliza o CrossEncoder local para reranking.

    Compatível com import do Jupyter:
        from src.vectorstore.hybrid_search import configurar_buscador
    """
    global _qdrant_client_singleton
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    pasta_processados = BASE_DIR / "data" / "processed"
    pasta_qdrant = BASE_DIR / "data" / "vectorstore"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": device},
    )

    # Carregar modelo BM25 salvo pelo embed.py
    caminho_bm25 = pasta_qdrant / "bm25.pkl"
    with open(caminho_bm25, "rb") as f:
        bm25 = pickle.load(f)

    # Inicializar CrossEncoder localmente para respeitar cibersegurança e privacidade de dados.
    # Modelo BAAI/bge-reranker-v2-m3: reranker multilíngue estado da arte, com forte
    # desempenho em textos técnicos/normativos em português (superior ao mmarco-mMiniLMv2
    # que foi treinado apenas em MS MARCO / inglês).
    cross_encoder = None
    if use_reranker:
        try:
            from sentence_transformers import CrossEncoder
            cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
        except ImportError:
            print("Aviso: sentence-transformers não está instalado. Reranking desativado.")

    # Prioridade de conexão ao Qdrant:
    # 1. Servidor Qdrant HTTP (Docker/externo via QDRANT_URL) — sem concorrência de lock de arquivos
    # 2. Qdrant local embutido (path=data/vectorstore) — fallback
    cliente = None
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        try:
            c_temp = QdrantClient(url=qdrant_url, timeout=3)
            colecoes = [c.name for c in c_temp.get_collections().collections]
            if "prodist_normativas" in colecoes:
                cliente = c_temp
        except Exception:
            cliente = None

    if cliente is None:
        # Qdrant embutido (sem Docker) — lê/escreve em data/vectorstore/
        # Usa singleton para evitar lock 'already accessed by another instance' no pytest
        if _qdrant_client_singleton is None:
            _qdrant_client_singleton = QdrantClient(path=str(pasta_qdrant))
        cliente = _qdrant_client_singleton

    # Carregar corpus para montar Documents a partir dos IDs retornados pelo RRF
    corpus_texts, corpus_metadados = _carregar_corpus(pasta_processados)

    return HybridRetriever(
        client=cliente,
        embeddings_model=embeddings,
        bm25=bm25,
        corpus_texts=corpus_texts,
        corpus_metadados=corpus_metadados,
        k=k_resultados,
        k_candidatos=k_candidatos,
        k_rrf=k_rrf,
        cross_encoder=cross_encoder,
        collection_name="prodist_normativas",
    )


if __name__ == "__main__":
    import sys

    print("Inicializando Hybrid Retriever (Dense + BM25 + RRF)...")
    try:
        buscador = configurar_buscador(k_resultados=5, k_candidatos=20)
    except Exception as e:
        print(f"Erro ao inicializar: {e}")
        print("Certifique-se de ter executado: python src/vectorstore/embed.py")
        sys.exit(1)

    print("Retriever pronto!\n")

    try:
        pergunta = input("Sua pergunta: ")
    except EOFError:
        sys.exit(0)
        
    if not pergunta.strip() or pergunta.strip().lower() in ["sair", "exit", "quit"]:
        sys.exit(0)

    print("=" * 70)
    print(f"QUERY: {pergunta}")
    print("-" * 70)

    # --- Dense separado ---
    dense_raw = buscador._busca_dense(pergunta)
    print(f"  [Dense top-3]")
    for rank, (cid, score) in enumerate(dense_raw[:3], 1):
        meta = buscador.corpus_metadados[cid]
        try:
            print(
                f"    #{rank} id={cid} score={score:.4f} "
                f"{meta.get('modulo','?')} Pag.{meta.get('pagina','?')}: "
                f"{buscador.corpus_texts[cid][:80].replace(chr(10),' ')}..."
            )
        except UnicodeEncodeError:
            print(f"    #{rank} id={cid} score={score:.4f} [Conteúdo contém caracteres não suportados pelo terminal]")

    # --- BM25 separado ---
    bm25_raw = buscador._busca_bm25(pergunta)
    print(f"  [BM25 top-3]")
    for rank, (cid, score) in enumerate(bm25_raw[:3], 1):
        meta = buscador.corpus_metadados[cid]
        try:
            print(
                f"    #{rank} id={cid} score={score:.4f} "
                f"{meta.get('modulo','?')} Pag.{meta.get('pagina','?')}: "
                f"{buscador.corpus_texts[cid][:80].replace(chr(10),' ')}..."
            )
        except UnicodeEncodeError:
            print(f"    #{rank} id={cid} score={score:.4f} [Conteúdo contém caracteres não suportados pelo terminal]")

    # --- Hybrid RRF ---
    docs = buscador.invoke(pergunta)
    print(f"  [Hybrid RRF top-{len(docs)}]")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        try:
            print(
                f"    #{i} {meta.get('modulo','?')} Pag.{meta.get('pagina','?')}: "
                f"{doc.page_content[:120].replace(chr(10),' ')}..."
            )
        except UnicodeEncodeError:
            print(f"    #{i} [Conteúdo contém caracteres não suportados pelo terminal]")
    print()