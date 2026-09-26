"""
tests/evaluation/run_all.py

Orquestrador da avaliação completa do RAG Híbrido.

Executa:
  1. Testes de ingestão (determinísticos)
  2. Avaliação de retrieval híbrido (Recall@K, Precision@K, MRR, NDCG)
  3. Avaliação de reranking (MRR antes/depois do reranking via RRF)
  4. Avaliação mínima de geração (N=3, minimiza chamadas à API)
  5. Geração do relatório consolidado (report.json + report.md)

Uso:
    python tests/evaluation/run_all.py

O pytest é invocado internamente. O relatório é salvo em:
    tests/evaluation/reports/report.json
    tests/evaluation/reports/report.md
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT_DIR / "tests" / "evaluation" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Garante imports do projeto
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ---------------------------------------------------------------------------
# Execução dos testes via pytest
# ---------------------------------------------------------------------------

def _run_pytest(test_paths: list[str], extra_args: list[str] = None) -> dict:
    """
    Executa pytest nos caminhos indicados e retorna estatísticas básicas.
    Retorna {'passed': int, 'failed': int, 'exit_code': int}.
    """
    import pytest

    args = test_paths + [
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-W", "ignore",
        "--no-header",
    ]
    if extra_args:
        args.extend(extra_args)

    # Plugin para capturar estatísticas
    class _StatPlugin:
        def __init__(self):
            self.passed = 0
            self.failed = 0
            self.skipped = 0
            self.errors = []

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                if report.passed:
                    self.passed += 1
                elif report.failed:
                    self.failed += 1
                    self.errors.append(f"{report.nodeid}: {report.longrepr}")
            if report.when == "setup" and report.skipped:
                self.skipped += 1

    plugin = _StatPlugin()
    exit_code = pytest.main(args, plugins=[plugin])
    return {
        "passed": plugin.passed,
        "failed": plugin.failed,
        "skipped": plugin.skipped,
        "exit_code": exit_code,
        "errors": plugin.errors,
    }


# ---------------------------------------------------------------------------
# Leitura de relatórios parciais gerados pelos testes
# ---------------------------------------------------------------------------

def _load_report(filename: str) -> dict:
    """Carrega relatório JSON gerado pelos testes ou retorna {} se não existir."""
    path = REPORTS_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _extract_retrieval_metrics(k: int = 5) -> dict:
    """Extrai métricas de retrieval do relatório full_metrics_k{k}.json."""
    data = _load_report(f"full_metrics_k{k}.json")
    if not data:
        return {}
    return {
        f"recall_at_{k}": data.get(f"context_recall@{k}"),
        f"precision_at_{k}": data.get(f"context_precision@{k}"),
        f"mrr_at_{k}": data.get(f"mrr@{k}"),
        f"ndcg_at_{k}": data.get(f"ndcg@{k}"),
        f"hit_rate_at_{k}": data.get(f"hit_rate@{k}"),
        "n_queries": data.get("n_queries"),
    }


def _extract_reranking_metrics() -> dict:
    """
    Extrai métricas de reranking (antes vs depois do RRF).
    O HybridRetriever usa RRF como mecanismo de reranking dos resultados
    Dense + BM25. Comparamos Dense isolado vs Hybrid (pós-RRF).
    """
    dense_k5 = _load_report("comparacao_retrievers_k5.json")
    hybrid_k5 = _load_report("full_metrics_k5.json")

    before = {}
    after = {}

    if dense_k5 and "Dense" in dense_k5:
        d = dense_k5["Dense"]
        before = {
            "mrr_at_5": d.get("mrr@5"),
            "ndcg_at_5": d.get("ndcg@5"),
            "hit_rate_at_5": d.get("hit_rate@5"),
            "nota": "Dense isolado (sem RRF)",
        }

    if hybrid_k5:
        after = {
            "mrr_at_5": hybrid_k5.get("mrr@5"),
            "ndcg_at_5": hybrid_k5.get("ndcg@5"),
            "hit_rate_at_5": hybrid_k5.get("hit_rate@5"),
            "nota": "Hybrid (Dense + BM25 + RRF)",
        }

    return {"before": before, "after": after}


def _extract_crossencoder_metrics() -> dict:
    """
    Extrai métricas de comparação RRF vs RRF + Cross-Encoder
    a partir do reranker_comparison.json gerado por eval_reranker.py.

    Returns:
        Dict com métricas do baseline e do Cross-Encoder para K=1, 3 e 5,
        ou dict vazio se o relatório ainda não foi gerado.
    """
    data = _load_report("reranker_comparison.json")
    if not data or "metricas" not in data:
        return {}

    metricas = data["metricas"]
    baseline = metricas.get("RRF (baseline)", {})
    reranker = metricas.get("RRF + Cross-Encoder", {})

    def _k(fonte: dict, k: int) -> dict:
        bloco = fonte.get(str(k), {})
        return {
            f"hit_rate_at_{k}": bloco.get(f"hit_rate@{k}"),
            f"mrr_at_{k}":      bloco.get(f"mrr@{k}"),
            f"ndcg_at_{k}":     bloco.get(f"ndcg@{k}"),
            f"cp_at_{k}":       bloco.get(f"context_precision@{k}"),
        }

    return {
        "config": data.get("config", {}),
        "baseline": {
            "k1": _k(baseline, 1),
            "k3": _k(baseline, 3),
            "k5": _k(baseline, 5),
            "nota": "RRF puro (sem Cross-Encoder)",
        },
        "cross_encoder": {
            "k1": _k(reranker, 1),
            "k3": _k(reranker, 3),
            "k5": _k(reranker, 5),
            "nota": "RRF + Cross-Encoder (BAAI/bge-reranker-v2-m3)",
        },
    }



def _extract_generation_metrics() -> dict:
    """Extrai métricas de geração do generation_suite.json."""
    data = _load_report("generation_suite.json")
    if not data:
        return {
            "evaluated_questions": None,
            "results": None,
            "nota": "generation_suite.json não encontrado — execute os testes de geração",
        }
    return {
        "evaluated_questions": data.get("n_perguntas_avaliadas"),
        "llm_calls_geracao": data.get("llm_calls_geracao"),
        "llm_calls_ragas": data.get("llm_calls_ragas", 0),
        "results": {
            "grounding_determinístico": data.get("grounding_determinístico"),
            "ragas": data.get("ragas"),
        },
    }


# ---------------------------------------------------------------------------
# Contagem de chamadas LLM estimadas
# ---------------------------------------------------------------------------

def _count_llm_calls(test_stats: dict, generation_metrics: dict) -> int:
    """
    Estima o número de chamadas à API LLM.
    - Geração: N_AMOSTRAS chamadas (1 por pergunta)
    - Ragas: N_AMOSTRAS × 2 (Faithfulness + Answer Relevancy)
    - Retrieval/Ingestão: 0 (determinísticos)
    """
    geracao = generation_metrics.get("llm_calls_geracao") or 0
    ragas = generation_metrics.get("llm_calls_ragas") or 0
    return geracao + ragas


# ---------------------------------------------------------------------------
# Geração do relatório Markdown
# ---------------------------------------------------------------------------

def _format_metric(value, fmt=".3f", null_str="null") -> str:
    if value is None:
        return null_str
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def _generate_markdown_report(report: dict) -> str:
    """Gera relatório Markdown a partir do dicionário de métricas."""
    s = report.get("summary", {})
    r5 = report.get("retrieval", {}).get("k5", {})
    r10 = report.get("retrieval", {}).get("k10", {})
    rerank = report.get("reranking", {})
    ce = report.get("cross_encoder", {})
    gen = report.get("generation", {})

    lines = [
        "# RAG Hybrid Evaluation Report",
        "",
        "## Resumo",
        f"- **Perguntas avaliadas (retrieval)**: {s.get('total_questions_retrieval', 'null')}",
        f"- **Perguntas avaliadas (geração)**: {s.get('total_questions_generation', 'null')}",
        f"- **Testes aprovados**: {s.get('tests_passed', 'null')}",
        f"- **Testes reprovados**: {s.get('tests_failed', 'null')}",
        f"- **Testes ignorados (skip)**: {s.get('tests_skipped', 'null')}",
        f"- **Tempo de execução**: {s.get('execution_time_seconds', 'null')}s",
        f"- **Chamadas ao LLM/API**: {s.get('llm_calls', 'null')}",
        "",
        "## Retrieval Híbrido",
        "",
        "| Métrica | K=5 | K=10 |",
        "|---------|-----|------|",
        f"| Recall@K | {_format_metric(r5.get('recall_at_5'))} | {_format_metric(r10.get('recall_at_10'))} |",
        f"| Precision@K | {_format_metric(r5.get('precision_at_5'))} | {_format_metric(r10.get('precision_at_10'))} |",
        f"| MRR@K | {_format_metric(r5.get('mrr_at_5'))} | {_format_metric(r10.get('mrr_at_10'))} |",
        f"| NDCG@K | {_format_metric(r5.get('ndcg_at_5'))} | {_format_metric(r10.get('ndcg_at_10'))} |",
        f"| Hit Rate@K | {_format_metric(r5.get('hit_rate_at_5'))} | {_format_metric(r10.get('hit_rate_at_10'))} |",
        "",
        "> **Nota sobre Recall@K**: Calculado como fração de chunks do módulo correto",
        "> recuperados em top-K sobre o total de chunks daquele módulo no corpus.",
        "> Recall real (não aproximado) quando corpus_por_modulo é fornecido.",
        "",
        "## Reranking (Dense → Hybrid RRF)",
        "",
        "O HybridRetriever aplica RRF como mecanismo de reranking dos resultados",
        "Dense (Qdrant) + BM25. A comparação abaixo mostra o ganho do RRF.",
        "",
        "| Métrica | Antes (Dense) | Depois (Hybrid RRF) |",
        "|---------|--------------|---------------------|",
    ]

    before = rerank.get("before", {})
    after = rerank.get("after", {})
    lines += [
        f"| MRR@5 | {_format_metric(before.get('mrr_at_5'))} | {_format_metric(after.get('mrr_at_5'))} |",
        f"| NDCG@5 | {_format_metric(before.get('ndcg_at_5'))} | {_format_metric(after.get('ndcg_at_5'))} |",
        f"| Hit Rate@5 | {_format_metric(before.get('hit_rate_at_5'))} | {_format_metric(after.get('hit_rate_at_5'))} |",
        "",
    ]

    # --- Seção Cross-Encoder ---
    lines.append("## Cross-Encoder Reranker (RRF → RRF + Cross-Encoder)")
    lines.append("")
    if ce:
        cfg = ce.get("config", {})
        n_q = cfg.get("n_perguntas", cfg.get("n_queries", "?"))
        modelo = ce.get("cross_encoder", {}).get("nota", "BAAI/bge-reranker-v2-m3")
        lines += [
            f"Modelo: `{modelo}` | Perguntas avaliadas: {n_q} | "
            f"k_rrf={cfg.get('k_rrf', '?')} candidatos → top-{cfg.get('k_resultados', '?')} final",
            "",
            "| Métrica | K | RRF Baseline | RRF + Cross-Encoder | Delta |",
            "|---------|---|-------------:|--------------------:|------:|",
        ]
        bl = ce.get("baseline", {})
        cr = ce.get("cross_encoder", {})
        for k in [1, 3, 5]:
            for metrica_label, campo in [("Hit@K", f"hit_rate_at_{k}"), ("MRR@K", f"mrr_at_{k}"), ("NDCG@K", f"ndcg_at_{k}")]:
                v_bl = bl.get(f"k{k}", {}).get(campo)
                v_cr = cr.get(f"k{k}", {}).get(campo)
                delta = ""
                if v_bl is not None and v_cr is not None:
                    d = v_cr - v_bl
                    delta = f"+{d:.3f}" if d >= 0 else f"{d:.3f}"
                lines.append(
                    f"| {metrica_label} | {k} | {_format_metric(v_bl)} | {_format_metric(v_cr)} | {delta} |"
                )
        lines.append("")
        lines.append("> Avaliação executada separadamente via `tests/evaluation/retrieval/eval_reranker.py`.")
    else:
        lines.append(
            "> _Avaliação de Cross-Encoder não encontrada. "
            "Execute `tests/evaluation/retrieval/eval_reranker.py` para gerar `reranker_comparison.json`._"
        )
    lines.append("")

    lines += [
        "## Geração",
        "",
    ]

    if gen.get("evaluated_questions"):
        lines.append(f"- **Perguntas avaliadas**: {gen['evaluated_questions']}")
        lines.append(f"- **Chamadas LLM (geração)**: {gen.get('llm_calls_geracao', 0)}")
        lines.append(f"- **Chamadas LLM (Ragas)**: {gen.get('llm_calls_ragas', 0)}")

        results = gen.get("results", {})
        if results:
            det = results.get("grounding_determinístico", {})
            if det:
                lines.append(f"- **Grounding (determinístico)**: {_format_metric(det.get('score'))}")
            ragas = results.get("ragas", {})
            if ragas and "faithfulness" in ragas:
                lines.append(f"- **Faithfulness (Ragas)**: {_format_metric(ragas.get('faithfulness'))}")
                lines.append(f"- **Answer Relevancy (Ragas)**: {_format_metric(ragas.get('answer_relevancy'))}")
            elif ragas:
                lines.append(f"- **Ragas**: {ragas}")
    else:
        lines.append("- Avaliação de geração não executada (API key ausente ou testes pulados)")

    lines += [
        "",
        "---",
        "_Relatório gerado automaticamente por `tests/evaluation/run_all.py`_",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def main():
    t_inicio = time.perf_counter()

    print("=" * 50)
    print("RAG HYBRID EVALUATION")
    print("=" * 50)

    # --- 1. Testes de Ingestão ---
    print("\n[1/3] Executando testes de ingestão...")
    ingestion_stats = _run_pytest([
        str(ROOT_DIR / "tests" / "test_ingestion.py"),
    ])

    # --- 2. Avaliação de Retrieval (testes determinísticos — sem LLM) ---
    print("\n[2/3] Executando avaliação de retrieval (sem LLM)...")
    retrieval_stats = _run_pytest([
        str(ROOT_DIR / "tests" / "evaluation" / "retrieval"),
    ])

    # --- 3. Avaliação de Geração (mínima — N=3, minimiza chamadas à API) ---
    print("\n[3/3] Executando avaliação de geração (N=3)...")
    generation_stats = _run_pytest([
        str(ROOT_DIR / "tests" / "evaluation" / "generation"),
    ])

    t_fim = time.perf_counter()
    execution_time = round(t_fim - t_inicio, 1)

    # --- Agregar estatísticas de testes ---
    total_passed = (
        ingestion_stats["passed"]
        + retrieval_stats["passed"]
        + generation_stats["passed"]
    )
    total_failed = (
        ingestion_stats["failed"]
        + retrieval_stats["failed"]
        + generation_stats["failed"]
    )
    total_skipped = (
        ingestion_stats["skipped"]
        + retrieval_stats["skipped"]
        + generation_stats["skipped"]
    )

    # --- Coletar métricas dos relatórios parciais ---
    retrieval_k5 = _extract_retrieval_metrics(k=5)
    retrieval_k10 = _extract_retrieval_metrics(k=10)
    reranking = _extract_reranking_metrics()
    crossencoder = _extract_crossencoder_metrics()
    generation = _extract_generation_metrics()
    llm_calls = _count_llm_calls(
        {"passed": total_passed, "failed": total_failed},
        generation,
    )

    # Número de perguntas no RETRIEVAL_SET
    try:
        from tests.evaluation.eval_dataset_v2 import RETRIEVAL_SET
        n_questions_retrieval = len(RETRIEVAL_SET)
    except Exception:
        n_questions_retrieval = None

    # --- Montar relatório consolidado ---
    report = {
        "summary": {
            "total_questions_retrieval": n_questions_retrieval,
            "total_questions_generation": generation.get("evaluated_questions"),
            "tests_passed": total_passed,
            "tests_failed": total_failed,
            "tests_skipped": total_skipped,
            "execution_time_seconds": execution_time,
            "llm_calls": llm_calls,
        },
        "retrieval": {
            "k5": retrieval_k5,
            "k10": retrieval_k10,
        },
        "reranking": reranking,
        "cross_encoder": crossencoder,
        "generation": generation,
        "notas": {
            "recall_at_k": (
                "Calculado como Context Recall (RAGAS): fração dos chunks do módulo correto "
                "recuperados em top-K dividida pelo total de chunks daquele módulo no corpus."
            ),
            "precision_at_k": (
                "Context Precision (RAGAS): fração ponderada de posições top-K contendo "
                "chunks relevantes (chunks relevantes no topo têm mais peso)."
            ),
            "reranking": (
                "O HybridRetriever usa RRF (Reciprocal Rank Fusion) como reranking. "
                "'before' = Dense isolado (sem RRF), 'after' = Hybrid (Dense+BM25+RRF)."
            ),
            "geração": (
                "Avaliação mínima com N=3 perguntas para economizar chamadas à API. "
                "Ragas (Faithfulness + Answer Relevancy) executado opcionalmente se API disponível."
            ),
        },
        "breakdowns": {
            "ingestion_tests": ingestion_stats,
            "retrieval_tests": retrieval_stats,
            "generation_tests": generation_stats,
        },
    }

    # --- Salvar relatórios ---
    report_json_path = REPORTS_DIR / "report.json"
    report_md_path = REPORTS_DIR / "report.md"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    report_md_path.write_text(
        _generate_markdown_report(report), encoding="utf-8"
    )

    # --- Mostrar resumo no terminal ---
    r5 = retrieval_k5
    r10 = retrieval_k10
    before = reranking.get("before", {})
    after = reranking.get("after", {})

    def fmt(v):
        return f"{v:.3f}" if v is not None else "null"

    print("\n")
    print("=" * 50)
    print("RAG HYBRID EVALUATION")
    print("=" * 50)
    print(f"\nQuestions evaluated: {n_questions_retrieval}")
    print("\nRetrieval")
    print(f"  Recall@5:   {fmt(r5.get('recall_at_5'))}")
    print(f"  Recall@10:  {fmt(r10.get('recall_at_10'))}")
    print(f"  Precision@5:{fmt(r5.get('precision_at_5'))}")
    print(f"  MRR@5:      {fmt(r5.get('mrr_at_5'))}")
    print(f"  NDCG@5:     {fmt(r5.get('ndcg_at_5'))}")

    print("\nReranking  (Dense → Hybrid RRF)")
    print(f"  MRR before: {fmt(before.get('mrr_at_5'))}")
    print(f"  MRR after:  {fmt(after.get('mrr_at_5'))}")

    print("\nGeneration")
    gen_q = generation.get("evaluated_questions")
    print(f"  Questions evaluated: {gen_q if gen_q else 'null (API key ausente)'}")
    ragas = (generation.get("results") or {}).get("ragas") or {}
    if "faithfulness" in ragas:
        print(f"  Faithfulness:  {fmt(ragas.get('faithfulness'))}")
        print(f"  Relevancy:     {fmt(ragas.get('answer_relevancy'))}")

    print("\nTests")
    print(f"  Passed:  {total_passed}")
    print(f"  Failed:  {total_failed}")
    print(f"  Skipped: {total_skipped}")

    if total_failed > 0:
        all_errors = (
            ingestion_stats.get("errors", [])
            + retrieval_stats.get("errors", [])
            + generation_stats.get("errors", [])
        )
        for err in all_errors[:5]:
            print(f"  ❌ {err[:100]}")

    print(f"\nLLM API calls: {llm_calls}")
    print(f"Execution time: {execution_time}s")
    print(f"\nReport:")
    print(f"  {report_json_path}")
    print(f"  {report_md_path}")
    print("\n" + "=" * 50)

    # Exit code: 0 se nenhum teste falhou, 1 caso contrário
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
