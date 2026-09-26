"""
tests/evaluation/retrieval/test_retrieval.py

Avalia:
  ETAPA 4 — Retrieval Evaluation
      Métricas obrigatórias RAG Híbrido:
        - Hit@K           : ao menos 1 resultado relevante em top-K
        - Context Precision@K (CP@K) : proporção ponderada de posições relevantes
        - Context Recall@K    (CR@K) : fração dos chunks relevantes recuperados
        - MRR@K           : Mean Reciprocal Rank
        - NDCG@K          : Normalized Discounted Cumulative Gain
  ETAPA 5 — Comparação Dense vs BM25 vs Hybrid
  ETAPA 6 — Teste de K (K=1,3,5,10,20) com análise de latência

IDENTIFICAÇÃO DE RELEVÂNCIA:
  Sem chunk_id nos metadados, a relevância é avaliada em nível de MÓDULO.
  hit = ao menos 1 chunk do módulo esperado foi recuperado em top-K.

  Quando a pergunta tem 'pagina_esperada' também calculamos métricas por (módulo, página).
  Essas métricas por página são mais rigorosas e registradas separadamente.

COMO FUNCIONA A SEPARAÇÃO Dense / BM25 / Hybrid:
  - Dense isolado: chama _busca_dense() interno e mapeia IDs → metadados do corpus
  - BM25 isolado:  chama _busca_bm25() interno e mapeia IDs → metadados do corpus
  - Hybrid:        invoca o retriever completo (Dense + BM25 + RRF)

  NÃO modificamos o retriever. Chamamos os métodos internos já públicos.
"""
import json
import time
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
# ---------------------------------------------------------------------------

from tests.evaluation.eval_dataset_v2 import RETRIEVAL_SET
from tests.evaluation.retrieval.metrics import (
    compute_all_k,
    compute_metrics,
    context_precision_at_k,
    context_recall_at_k,
    mrr_at_k,
    formatar_tabela_metricas,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _ids_to_metadados(ids: list[int], corpus_metadados: list[dict]) -> list[dict]:
    """Converte lista de chunk_ids (índices inteiros) em lista de metadados."""
    result = []
    for cid in ids:
        if 0 <= cid < len(corpus_metadados):
            result.append(corpus_metadados[cid])
    return result


def _dense_retrieve(query: str, k: int, components: dict) -> list[dict]:
    """
    Executa Dense retrieval isolado e retorna metadados dos top-K chunks.
    Usa _busca_dense() do HybridRetriever sem modificá-lo.
    """
    retriever = components["retriever"]
    # Temporariamente ajustamos k_candidatos para usar o k desejado
    original_k_cand = retriever.k_candidatos
    retriever.k_candidatos = max(k, 20)  # garantir candidatos suficientes

    try:
        dense_results = retriever._busca_dense(query)[:k]
    finally:
        retriever.k_candidatos = original_k_cand

    ids = [cid for cid, _ in dense_results]
    return _ids_to_metadados(ids, components["corpus_metadados"])


def _bm25_retrieve(query: str, k: int, components: dict) -> list[dict]:
    """
    Executa BM25 retrieval isolado e retorna metadados dos top-K chunks.
    """
    retriever = components["retriever"]
    original_k_cand = retriever.k_candidatos
    retriever.k_candidatos = max(k, 20)

    try:
        bm25_results = retriever._busca_bm25(query)[:k]
    finally:
        retriever.k_candidatos = original_k_cand

    ids = [cid for cid, _ in bm25_results]
    return _ids_to_metadados(ids, components["corpus_metadados"])


def _hybrid_retrieve(query: str, k: int, retriever_components: dict) -> list[dict]:
    """
    Executa Hybrid retrieval e retorna metadados dos top-K documentos.
    Cria instância temporária com o k desejado para não modificar o original.
    """
    from src.vectorstore.hybrid_search import HybridRetriever
    comps = retriever_components

    temp_retriever = HybridRetriever(
        client=comps["client"],
        embeddings_model=comps["embeddings"],
        bm25=comps["bm25"],
        corpus_texts=comps["corpus_texts"],
        corpus_metadados=comps["corpus_metadados"],
        collection_name=comps["collection_name"],
        k=k,
        k_candidatos=max(k * 2, 20),
    )
    docs = temp_retriever.invoke(query)
    return [doc.metadata for doc in docs]


# ---------------------------------------------------------------------------
# Etapa 4 — Retrieval Evaluation (Hybrid, K=5 e K=10)
# ---------------------------------------------------------------------------

class TestHybridRetrieval:
    """Avalia o HybridRetriever padrão nos K usados em produção."""

    def test_hybrid_hit_rate_k5(self, retriever_components, corpus, reports_dir):
        """Hit@5: ao menos 1 chunk do módulo correto em top-5."""
        _, corpus_metadados = corpus
        comps = retriever_components

        resultados = []
        for entrada in RETRIEVAL_SET:
            docs = _hybrid_retrieve(entrada["pergunta"], k=5, retriever_components=comps)
            resultados.append({
                "id": entrada["id"],
                "docs_metadados": docs,
                "modulo_esperado": entrada["modulo_esperado"],
            })

        metricas = compute_metrics(resultados, k=5)

        # Salvar resultado
        _salvar_resultado(reports_dir, "hybrid_k5.json", metricas)

        print(f"\n[Hybrid] Hit@5={metricas['hit_rate@5']:.3f} | "
              f"Recall@5={metricas['recall@5']:.3f} | "
              f"MRR@5={metricas['mrr@5']:.3f} | "
              f"NDCG@5={metricas['ndcg@5']:.3f}")

        # Threshold mínimo esperado — registrar, não falhar
        # Comentamos o assert para não mascarar resultados ruins
        # assert metricas["hit_rate@5"] >= 0.0, "Métrica calculada"

    def test_hybrid_hit_rate_k10(self, retriever_components, corpus, reports_dir):
        """Hit@10: ao menos 1 chunk do módulo correto em top-10."""
        _, corpus_metadados = corpus
        comps = retriever_components

        resultados = []
        for entrada in RETRIEVAL_SET:
            docs = _hybrid_retrieve(entrada["pergunta"], k=10, retriever_components=comps)
            resultados.append({
                "id": entrada["id"],
                "docs_metadados": docs,
                "modulo_esperado": entrada["modulo_esperado"],
            })

        metricas = compute_metrics(resultados, k=10)
        _salvar_resultado(reports_dir, "hybrid_k10.json", metricas)

        print(f"\n[Hybrid] Hit@10={metricas['hit_rate@10']:.3f} | "
              f"Recall@10={metricas['recall@10']:.3f} | "
              f"MRR@10={metricas['mrr@10']:.3f} | "
              f"NDCG@10={metricas['ndcg@10']:.3f}")

    def test_hybrid_per_query_breakdown(self, retriever_components, corpus, reports_dir):
        """Detalha hits/misses por pergunta para diagnóstico qualitativo."""
        _, corpus_metadados = corpus
        comps = retriever_components
        k = 5

        report_linhas = ["# Retrieval — Detalhe por Pergunta (Hybrid, K=5)\n"]
        report_linhas.append("| ID | Hit@5 | Módulo Esperado | Módulos Recuperados |")
        report_linhas.append("|----|----|---|---|")

        for entrada in RETRIEVAL_SET:
            docs = _hybrid_retrieve(entrada["pergunta"], k=k, retriever_components=comps)
            hit = any(
                d.get("modulo") == entrada["modulo_esperado"] for d in docs
            )
            modulos_rec = list({d.get("modulo", "?") for d in docs})
            status = "✅" if hit else "❌"
            report_linhas.append(
                f"| {entrada['id']} | {status} | {entrada['modulo_esperado']} | "
                f"{', '.join(modulos_rec)} |"
            )

        report = "\n".join(report_linhas)
        (reports_dir / "hybrid_per_query.md").write_text(report, encoding="utf-8")
        print(f"\n{report}")


# ---------------------------------------------------------------------------
# Etapa 5 — Comparação Dense vs BM25 vs Hybrid
# ---------------------------------------------------------------------------

class TestRetrieverComparison:
    """Compara os três retrievers com o mesmo conjunto de perguntas."""

    @pytest.mark.parametrize("k", [5, 10])
    def test_comparison_all_retrievers(self, retriever_components, corpus, reports_dir, k):
        """
        Produz tabela comparativa:
        | Retriever | Hit@K | Recall@K | MRR@K | NDCG@K |
        """
        _, corpus_metadados = corpus
        comps = retriever_components

        todos_resultados = {"Dense": [], "BM25": [], "Hybrid": []}

        for entrada in RETRIEVAL_SET:
            q = entrada["pergunta"]
            mod = entrada["modulo_esperado"]

            todos_resultados["Dense"].append({
                "id": entrada["id"],
                "docs_metadados": _dense_retrieve(q, k, comps),
                "modulo_esperado": mod,
            })
            todos_resultados["BM25"].append({
                "id": entrada["id"],
                "docs_metadados": _bm25_retrieve(q, k, comps),
                "modulo_esperado": mod,
            })
            todos_resultados["Hybrid"].append({
                "id": entrada["id"],
                "docs_metadados": _hybrid_retrieve(q, k, comps),
                "modulo_esperado": mod,
            })

        tabela_data = {}
        for retriever_nome, resultados in todos_resultados.items():
            metricas = compute_metrics(resultados, k)
            tabela_data[retriever_nome] = {str(k): metricas}

        tabela = formatar_tabela_metricas(tabela_data)
        _salvar_resultado(reports_dir, f"comparacao_retrievers_k{k}.json", {
            nome: compute_metrics(res, k)
            for nome, res in todos_resultados.items()
        })
        (reports_dir / f"comparacao_retrievers_k{k}.md").write_text(
            f"# Comparação Retrievers @ K={k}\n\n{tabela}\n", encoding="utf-8"
        )

        print(f"\n{'='*60}")
        print(f"COMPARAÇÃO RETRIEVERS @ K={k}")
        print(tabela)
        print(f"{'='*60}")

    def test_bm25_advantage_exact_terms(self, retriever_components, corpus):
        """
        Verifica casos onde BM25 deve superar Dense:
        perguntas com siglas exatas (DIC, FEC, DEC).
        """
        comps = retriever_components
        sigla_queries = [
            ("Como é calculado o DIC?", "Módulo 8"),
            ("Quais os limites do FEC?", "Módulo 8"),
            ("Critério de cálculo do DEC segundo o PRODIST", "Módulo 8"),
        ]

        dense_hits, bm25_hits = 0, 0
        for query, mod_esp in sigla_queries:
            dense_docs = _dense_retrieve(query, 5, comps)
            bm25_docs = _bm25_retrieve(query, 5, comps)

            dense_hit = any(d.get("modulo") == mod_esp for d in dense_docs)
            bm25_hit = any(d.get("modulo") == mod_esp for d in bm25_docs)

            if dense_hit:
                dense_hits += 1
            if bm25_hit:
                bm25_hits += 1

            print(f"  Query: '{query}'")
            print(f"    Dense hit: {dense_hit} | BM25 hit: {bm25_hit}")

        print(f"\n  Dense hits com siglas: {dense_hits}/{len(sigla_queries)}")
        print(f"  BM25  hits com siglas: {bm25_hits}/{len(sigla_queries)}")


# ---------------------------------------------------------------------------
# Etapa 6 — Teste de K
# ---------------------------------------------------------------------------

class TestKValues:
    """Avalia o impacto de K=1,3,5,10,20 nas métricas e latência."""

    KS = [1, 3, 5, 10, 20]

    def test_k_sweep_hybrid(self, retriever_components, corpus, reports_dir):
        """
        Para cada K in [1,3,5,10,20]:
          - Computa métricas do Hybrid
          - Mede latência de retrieval
        """
        _, corpus_metadados = corpus
        comps = retriever_components

        resultados_por_k = {}
        latencias_por_k = {}

        for k in self.KS:
            resultados = []
            latencias = []

            for entrada in RETRIEVAL_SET:
                t0 = time.perf_counter()
                docs = _hybrid_retrieve(entrada["pergunta"], k, comps)
                lat = time.perf_counter() - t0

                latencias.append(lat)
                resultados.append({
                    "id": entrada["id"],
                    "docs_metadados": docs,
                    "modulo_esperado": entrada["modulo_esperado"],
                })

            metricas = compute_metrics(resultados, k)
            resultados_por_k[k] = metricas

            import statistics
            latencias_por_k[k] = {
                "mean_ms": statistics.mean(latencias) * 1000,
                "median_ms": statistics.median(latencias) * 1000,
                "p95_ms": _percentile(latencias, 95) * 1000,
            }

        # Exibir resultado
        print("\n" + "="*70)
        print("TESTE DE K — Hybrid Retriever")
        print(f"{'K':>4} | {'Hit@K':>7} | {'Recall@K':>9} | {'MRR@K':>7} | {'NDCG@K':>7} | {'Lat.Mean(ms)':>13}")
        print("-"*70)
        for k in self.KS:
            m = resultados_por_k[k]
            l = latencias_por_k[k]
            print(
                f"{k:>4} | {m[f'hit_rate@{k}']:>7.3f} | {m[f'recall@{k}']:>9.3f} | "
                f"{m[f'mrr@{k}']:>7.3f} | {m[f'ndcg@{k}']:>7.3f} | {l['mean_ms']:>13.1f}"
            )
        print("="*70)

        # Salvar
        _salvar_resultado(reports_dir, "k_sweep_hybrid.json", {
            "metricas": {str(k): resultados_por_k[k] for k in self.KS},
            "latencias": {str(k): latencias_por_k[k] for k in self.KS},
        })

    def test_k_sweep_dense_vs_bm25(self, retriever_components, corpus, reports_dir):
        """Compara Dense vs BM25 para K=5 e K=10 para validar ganho do Hybrid."""
        _, corpus_metadados = corpus
        comps = retriever_components

        report = {}
        for retriever_nome, fn in [("Dense", _dense_retrieve), ("BM25", _bm25_retrieve)]:
            por_k = {}
            for k in [5, 10]:
                resultados = []
                for entrada in RETRIEVAL_SET:
                    docs = fn(entrada["pergunta"], k, comps)
                    resultados.append({
                        "id": entrada["id"],
                        "docs_metadados": docs,
                        "modulo_esperado": entrada["modulo_esperado"],
                    })
                por_k[str(k)] = compute_metrics(resultados, k)
            report[retriever_nome] = por_k

        _salvar_resultado(reports_dir, "k_sweep_dense_bm25.json", report)
        print(f"\nResultados Dense vs BM25 por K salvos em reports/k_sweep_dense_bm25.json")


# ---------------------------------------------------------------------------
# ETAPA 4b — Context Precision, Context Recall e MRR explícitos
# ---------------------------------------------------------------------------

class TestContextPrecisionRecall:
    """
    Avalia as métricas obrigatórias do RAG Híbrido alinhadas com Ragas:

    - Context Precision@K (CP@K): fração ponderada de posições top-K que são relevantes.
      Chunks relevantes no topo têm mais peso (Ragas context_precision).

    - Context Recall@K (CR@K): fração dos chunks relevantes do ground truth
      que foram efetivamente recuperados em top-K (Ragas context_recall).

    - MRR@K: Mean Reciprocal Rank — quão cedo o primeiro chunk relevante aparece.
    """

    def test_context_precision_k5(self, retriever_components, reports_dir):
        """
        Context Precision@5: mede se os chunks relevantes ficam no topo do ranking.
        Um CP@5 alto indica que o retriever prioriza corretamente o módulo esperado.
        """
        comps = retriever_components
        resultados_cp = []

        for entrada in RETRIEVAL_SET:
            docs = _hybrid_retrieve(entrada["pergunta"], k=5, retriever_components=comps)
            cp = context_precision_at_k(docs, entrada["modulo_esperado"], k=5)
            resultados_cp.append({
                "id": entrada["id"],
                "context_precision@5": cp,
                "modulo_esperado": entrada["modulo_esperado"],
            })

        media_cp = sum(r["context_precision@5"] for r in resultados_cp) / max(len(resultados_cp), 1)

        _salvar_resultado(reports_dir, "context_precision_k5.json", {
            "media_context_precision@5": media_cp,
            "detalhes": resultados_cp,
        })

        print(f"\n[Hybrid] Context Precision@5 (média): {media_cp:.3f}")
        for r in resultados_cp:
            status = "✅" if r["context_precision@5"] > 0 else "❌"
            print(f"  {status} [{r['id']}] CP@5={r['context_precision@5']:.3f} mod={r['modulo_esperado']}")

    def test_context_recall_k5(self, retriever_components, corpus, reports_dir):
        """
        Context Recall@5: mede qual proporção do ground truth relevante foi recuperada.
        Usa o corpus para calcular o denominador real (total de chunks por módulo).
        """
        corpus_texts, corpus_metadados = corpus
        comps = retriever_components

        # Contar chunks por módulo no corpus para recall real
        from collections import Counter
        chunks_por_modulo: dict[str, int] = Counter(
            m.get("modulo", "") for m in corpus_metadados
        )

        resultados_cr = []

        for entrada in RETRIEVAL_SET:
            docs = _hybrid_retrieve(entrada["pergunta"], k=5, retriever_components=comps)
            total_rel = chunks_por_modulo.get(entrada["modulo_esperado"])
            cr = context_recall_at_k(docs, entrada["modulo_esperado"], k=5, total_relevantes=total_rel)
            resultados_cr.append({
                "id": entrada["id"],
                "context_recall@5": cr,
                "modulo_esperado": entrada["modulo_esperado"],
                "total_chunks_modulo": total_rel,
            })

        media_cr = sum(r["context_recall@5"] for r in resultados_cr) / max(len(resultados_cr), 1)

        _salvar_resultado(reports_dir, "context_recall_k5.json", {
            "media_context_recall@5": media_cr,
            "detalhes": resultados_cr,
        })

        print(f"\n[Hybrid] Context Recall@5 (média): {media_cr:.3f}")
        for r in resultados_cr:
            print(f"  [{r['id']}] CR@5={r['context_recall@5']:.3f} "
                  f"(total chunks módulo={r['total_chunks_modulo']})")

    def test_mrr_k5_and_k10(self, retriever_components, reports_dir):
        """
        MRR@5 e MRR@10: quão cedo o primeiro chunk relevante aparece no ranking.
        MRR@K = média de 1/rank_primeiro_relevante sobre todas as perguntas.
        """
        comps = retriever_components
        resultados_mrr = []

        for entrada in RETRIEVAL_SET:
            docs_k10 = _hybrid_retrieve(entrada["pergunta"], k=10, retriever_components=comps)
            rr5  = mrr_at_k(docs_k10, entrada["modulo_esperado"], k=5)
            rr10 = mrr_at_k(docs_k10, entrada["modulo_esperado"], k=10)
            resultados_mrr.append({
                "id": entrada["id"],
                "rr@5": rr5,
                "rr@10": rr10,
                "modulo_esperado": entrada["modulo_esperado"],
            })

        n = max(len(resultados_mrr), 1)
        mrr5  = sum(r["rr@5"]  for r in resultados_mrr) / n
        mrr10 = sum(r["rr@10"] for r in resultados_mrr) / n

        _salvar_resultado(reports_dir, "mrr_k5_k10.json", {
            "mrr@5": mrr5,
            "mrr@10": mrr10,
            "detalhes": resultados_mrr,
        })

        print(f"\n[Hybrid] MRR@5={mrr5:.3f} | MRR@10={mrr10:.3f}")

    def test_full_metrics_comparison_k5_k10(self, retriever_components, corpus, reports_dir):
        """
        Tabela completa com todas as métricas obrigatórias para K=5 e K=10:
        Hit@K | CP@K | CR@K | MRR@K | NDCG@K
        """
        corpus_texts, corpus_metadados = corpus
        comps = retriever_components

        from collections import Counter
        chunks_por_modulo: dict[str, int] = Counter(
            m.get("modulo", "") for m in corpus_metadados
        )

        for k in [5, 10]:
            resultados = []
            for entrada in RETRIEVAL_SET:
                docs = _hybrid_retrieve(entrada["pergunta"], k=k, retriever_components=comps)
                resultados.append({
                    "id": entrada["id"],
                    "docs_metadados": docs,
                    "modulo_esperado": entrada["modulo_esperado"],
                })

            metricas = compute_metrics(resultados, k=k, corpus_por_modulo=chunks_por_modulo)
            _salvar_resultado(reports_dir, f"full_metrics_k{k}.json", metricas)

            print(f"\n[Hybrid K={k}] "
                  f"Hit@{k}={metricas[f'hit_rate@{k}']:.3f} | "
                  f"CP@{k}={metricas[f'context_precision@{k}']:.3f} | "
                  f"CR@{k}={metricas[f'context_recall@{k}']:.3f} | "
                  f"MRR@{k}={metricas[f'mrr@{k}']:.3f} | "
                  f"NDCG@{k}={metricas[f'ndcg@{k}']:.3f}")


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _percentile(data: list[float], p: float) -> float:
    """Percentil p (0-100) de uma lista de floats."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_data) - 1)
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


def _salvar_resultado(reports_dir: Path, nome: str, dados: dict):
    """Salva resultado em JSON no diretório de relatórios."""
    import json
    caminho = reports_dir / nome
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
