# RAG Hybrid Evaluation Report

## Resumo
- **Perguntas avaliadas (retrieval)**: 34
- **Perguntas avaliadas (geração)**: 3
- **Testes aprovados**: 25
- **Testes reprovados**: 0
- **Testes ignorados (skip)**: 0
- **Tempo de execução**: 150.5s
- **Chamadas ao LLM/API**: 3

## Retrieval Híbrido

| Métrica | K=5 | K=10 |
|---------|-----|------|
| Recall@K | 0.020 | 0.036 |
| Precision@K | 0.622 | 0.581 |
| MRR@K | 0.633 | 0.647 |
| NDCG@K | 0.656 | 0.676 |
| Hit Rate@K | 0.735 | 0.853 |

> **Nota sobre Recall@K**: Calculado como fração de chunks do módulo correto
> recuperados em top-K sobre o total de chunks daquele módulo no corpus.
> Recall real (não aproximado) quando corpus_por_modulo é fornecido.

## Reranking (Dense → Hybrid RRF)

O HybridRetriever aplica RRF como mecanismo de reranking dos resultados
Dense (Qdrant) + BM25. A comparação abaixo mostra o ganho do RRF.

| Métrica | Antes (Dense) | Depois (Hybrid RRF) |
|---------|--------------|---------------------|
| MRR@5 | 0.535 | 0.633 |
| NDCG@5 | 0.575 | 0.656 |
| Hit Rate@5 | 0.706 | 0.735 |

## Cross-Encoder Reranker (RRF → RRF + Cross-Encoder)

Modelo: `RRF + Cross-Encoder (BAAI/bge-reranker-v2-m3)` | Perguntas avaliadas: 34 | k_rrf=20 candidatos → top-5 final

| Métrica | K | RRF Baseline | RRF + Cross-Encoder | Delta |
|---------|---|-------------:|--------------------:|------:|
| Hit@K | 1 | 0.559 | 0.588 | +0.029 |
| MRR@K | 1 | 0.559 | 0.588 | +0.029 |
| NDCG@K | 1 | 0.559 | 0.588 | +0.029 |
| Hit@K | 3 | 0.706 | 0.735 | +0.029 |
| MRR@K | 3 | 0.627 | 0.657 | +0.029 |
| NDCG@K | 3 | 0.647 | 0.680 | +0.034 |
| Hit@K | 5 | 0.735 | 0.735 | +0.000 |
| MRR@K | 5 | 0.633 | 0.657 | +0.024 |
| NDCG@K | 5 | 0.656 | 0.671 | +0.015 |

> Avaliação executada separadamente via `tests/evaluation/retrieval/eval_reranker.py`.

## Geração

- **Perguntas avaliadas**: 3
- **Chamadas LLM (geração)**: 3
- **Chamadas LLM (Ragas)**: 0
- **Grounding (determinístico)**: 1.000
- **Ragas**: {'erro': "Faithfulness.__init__() missing 1 required positional argument: 'llm'"}

---
_Relatório gerado automaticamente por `tests/evaluation/run_all.py`_