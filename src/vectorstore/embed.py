"""
Etapa 2 — Geração de embeddings e indexação no Qdrant.

Indexa SOMENTE vetor Dense por chunk:
  - Dense (384d): HuggingFace paraphrase-multilingual-MiniLM-L12-v2

O BM25 é construído separadamente e salvo em data/vectorstore/bm25.pkl
para uso no hybrid_search.py (busca lexical em memória).

A coleção é SEMPRE recriada do zero ao executar este script,
garantindo que execuções repetidas não gerem duplicatas.
"""
import json
import os
import pickle
import re
from pathlib import Path

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi


def carregar_chunks(caminho_arquivo: str) -> list:
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenizar_pt(texto: str) -> list[str]:
    """
    Tokenização básica para português.
    DEVE ser idêntica à usada em hybrid_search.py para garantir
    que o índice BM25 e as queries usem o mesmo vocabulário.
    """
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", " ", texto)
    return [t for t in texto.split() if len(t) > 2]


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    pasta_processados = BASE_DIR / "data" / "processed"
    pasta_qdrant = BASE_DIR / "data" / "vectorstore"
    os.makedirs(pasta_qdrant, exist_ok=True)

    # ── Modelo de Embeddings ────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregando modelo de Embeddings (dispositivo: {device})...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": device},
    )

    # ── Carregar todos os chunks (ORDEM DEVE SER IDÊNTICA À DO hybrid_search.py) ──
    # Usa sorted() para garantir ordem determinística e reproduzível.
    arquivos_chunks = sorted(pasta_processados.glob("*_chunks.json"))
    if not arquivos_chunks:
        print("Nenhum arquivo de chunks encontrado em data/processed/.")
    else:
        todos_os_chunks: list[dict] = []
        for arquivo in arquivos_chunks:
            print(f"  Lendo: {arquivo.name}")
            todos_os_chunks.extend(carregar_chunks(arquivo))

        print(f"\nTotal de chunks: {len(todos_os_chunks)}")

        textos = [c["texto"] for c in todos_os_chunks]
        metadados = [c["metadados"] for c in todos_os_chunks]

        # ── Índice BM25 ─────────────────────────────────────────────────────────
        # O BM25 é salvo em disco para ser carregado pelo hybrid_search.py.
        # A ordem dos textos aqui DEVE ser idêntica à ordem de carregamento
        # no hybrid_search.py para que os índices retornados sejam válidos.
        print("\nConstruindo índice BM25...")
        tokenized_corpus = [tokenizar_pt(t) for t in textos]
        bm25 = BM25Okapi(tokenized_corpus)
        print(f"  BM25 pronto — corpus: {len(textos)} documentos.")

        caminho_bm25 = pasta_qdrant / "bm25.pkl"
        with open(caminho_bm25, "wb") as f:
            pickle.dump(bm25, f)
        print(f"  Modelo BM25 salvo em {caminho_bm25.name}")

        # ── Embeddings Densos ────────────────────────────────────────────────────
        print("\nGerando embeddings densos (pode demorar)...")
        dense_vectors = embeddings_model.embed_documents(textos)
        dim = len(dense_vectors[0])
        print(f"  Embeddings gerados: {len(dense_vectors)} x {dim} dims.")

        # ── Qdrant — recriar coleção SOMENTE DENSE ───────────────────────────────
        print("\nRecriando coleção no Qdrant (limpando diretorio)...")
        import shutil
        # Força a limpeza física do Qdrant local para evitar ghost points
        if pasta_qdrant.exists():
            try:
                shutil.rmtree(pasta_qdrant)
            except Exception as e:
                print(f"Aviso: Não foi possível limpar o diretório {pasta_qdrant}: {e}")
                
        os.makedirs(pasta_qdrant, exist_ok=True)
        
        # O bm25.pkl acabou de ser apagado se estava lá dentro, então salvamos de novo
        with open(caminho_bm25, "wb") as f:
            pickle.dump(bm25, f)
            
        # Qdrant embutido (sem Docker) — persiste em data/vectorstore/
        # Para usar servidor externo, troque por: QdrantClient(url=os.getenv("QDRANT_URL"))
        cliente = QdrantClient(path=str(pasta_qdrant))
        try:
            try:
                cliente.delete_collection("prodist_normativas")
                print("  Coleção antiga removida via API.")
            except Exception:
                pass

            # Colecao com APENAS vetor dense — sem sparse_vectors_config
            cliente.create_collection(
                collection_name="prodist_normativas",
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            print(f"  Coleção criada: vetores densos ({dim}d, Cosine).")
    
            # ── Upsert em lotes ──────────────────────────────────────────────────
            # IDs são inteiros sequenciais: id=i corresponde ao chunk i do corpus.
            # Isso permite que o hybrid_search.py use o ID do ponto Qdrant como
            # índice direto no BM25 para recuperar o payload correto.
            print(f"\nIndexando {len(textos)} chunks no Qdrant...")
            pontos = [
                PointStruct(
                    id=i,
                    vector=dense,
                    payload={"page_content": texto, "metadata": meta},
                )
                for i, (texto, meta, dense) in enumerate(
                    zip(textos, metadados, dense_vectors)
                )
            ]

            BATCH = 100
            total_lotes = (len(pontos) - 1) // BATCH + 1
            for start in range(0, len(pontos), BATCH):
                lote = pontos[start: start + BATCH]
                cliente.upsert("prodist_normativas", points=lote)
                print(
                    f"  Lote {start // BATCH + 1}/{total_lotes} "
                    f"({len(lote)} chunks) OK"
                )

            info = cliente.get_collection("prodist_normativas")
            print(f"\nColeção pronta: {info.points_count} pontos indexados.")
            print("Todos os documentos foram vetorizados e o Banco Vetorial esta pronto!")
        finally:
            cliente.close()  # evita o warning QdrantClient.__del__