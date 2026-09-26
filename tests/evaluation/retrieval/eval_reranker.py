"""
tests/evaluation/retrieval/eval_reranker.py

Avaliação quantitativa do impacto do Cross-Encoder no pipeline de retrieval.

Compara dois retrievers usando as métricas já definidas em metrics.py:
  - Retriever A: apenas RRF (baseline)
  - Retriever B: RRF + Cross-Encoder (experimental)

Métricas avaliadas em K = {1, 3, 5}:
  Hit@K | Context Precision@K | Context Recall@K | MRR@K | NDCG@K

Dataset: perguntas REGULATORIO do eval_dataset_v2.py (únicas com modulo_esperado definido,
         necessário para calcular relevância por módulo PRODIST).

Uso:
    .venv\\Scripts\\python.exe tests\\evaluation\\retrieval\\eval_reranker.py

Saída:
  - Tabela markdown comparativa no terminal
  - Arquivo JSON em tests/evaluation/reports/reranker_comparison.json
  - Arquivo Markdown em tests/evaluation/reports/reranker_comparison.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Garante que o root do projeto está no path
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from tests.evaluation.eval_dataset_v2 import EVAL_SET_V2
from tests.evaluation.retrieval.metrics import compute_all_k, formatar_tabela_metricas
from src.vectorstore.hybrid_search import configurar_buscador

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

KS = [1, 3, 5]
K_CANDIDATOS = 20
K_RRF = 20
K_FINAL = 5

REPORTS_DIR = ROOT / "tests" / "evaluation" / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Coleta de resultados
# ---------------------------------------------------------------------------

def coletar_resultados(retriever, perguntas: list[dict]) -> list[dict]:
    """
    Executa o retriever para cada pergunta e retorna lista de resultados
    no formato esperado por compute_metrics().

    Args:
        retriever: Instância do HybridRetriever (com ou sem Cross-Encoder).
        perguntas: Subconjunto do EVAL_SET com campo 'modulo_esperado' definido.

    Returns:
        Lista de dicts com 'docs_metadados' e 'modulo_esperado'.
    """
    resultados = []
    for item in perguntas:
        try:
            docs = retriever.invoke(item["pergunta"])
            resultados.append({
                "id": item["id"],
                "pergunta": item["pergunta"],
                "modulo_esperado": item["modulo_esperado"],
                "docs_metadados": [d.metadata for d in docs],
            })
        except Exception as e:
            print(f"  [ERRO] {item['id']}: {e}")
    return resultados


def avaliar_retriever(nome: str, retriever, perguntas: list[dict]) -> tuple[list[dict], dict]:
    """
    Avalia um retriever e retorna os resultados brutos e as métricas agregadas.

    Args:
        nome: Nome do retriever para exibição.
        retriever: Instância do HybridRetriever.
        perguntas: Lista de itens do dataset com 'modulo_esperado'.

    Returns:
        Tupla (resultados_brutos, metricas_por_k).
    """
    print(f"\n  Avaliando: {nome} ({len(perguntas)} perguntas)...")
    t0 = time.time()
    resultados = coletar_resultados(retriever, perguntas)
    elapsed = time.time() - t0
    metricas = compute_all_k(resultados, ks=KS)
    print(f"  Concluido em {elapsed:.1f}s")
    return resultados, metricas


# ---------------------------------------------------------------------------
# Geração de relatório
# ---------------------------------------------------------------------------

def gerar_markdown(
    metricas_por_retriever: dict[str, dict],
    resultados_por_retriever: dict[str, list[dict]],
    perguntas: list[dict],
) -> str:
    """
    Gera relatório markdown completo com tabela de métricas e análise por pergunta.

    Args:
        metricas_por_retriever: {nome: {k: {metrica: valor}}}
        resultados_por_retriever: {nome: [resultados_brutos]}
        perguntas: Lista de itens do dataset avaliados.

    Returns:
        String markdown com o relatório completo.
    """
    linhas = [
        "# Avaliacao de Reranking — RRF vs RRF + Cross-Encoder",
        "",
        "## Resumo das Metricas",
        "",
        formatar_tabela_metricas(metricas_por_retriever),
        "",
        "> **Metricas:** Hit@K = documento relevante encontrado no top-K | "
        "CP@K = Context Precision | CR@K = Context Recall | "
        "MRR@K = Mean Reciprocal Rank | NDCG@K = Normalized DCG",
        "",
        "## Analise por Pergunta (K=5)",
        "",
        "| ID | Pergunta | Modulo | RRF Hit | Reranker Hit | Mudou? |",
        "|----|----|----|----|----|----|",
    ]

    resultados_a = {r["id"]: r for r in resultados_por_retriever.get("RRF (baseline)", [])}
    resultados_b = {r["id"]: r for r in resultados_por_retriever.get("RRF + Cross-Encoder", [])}

    for item in perguntas:
        id_ = item["id"]
        mod = item["modulo_esperado"]
        k = 5

        res_a = resultados_a.get(id_)
        res_b = resultados_b.get(id_)

        hit_a = "-"
        hit_b = "-"
        mudou = "-"

        if res_a:
            mods_a = [m.get("modulo", "") for m in res_a["docs_metadados"][:k]]
            hit_a = "OK" if mod in mods_a else "MISS"

        if res_b:
            mods_b = [m.get("modulo", "") for m in res_b["docs_metadados"][:k]]
            hit_b = "OK" if mod in mods_b else "MISS"

        if res_a and res_b:
            ids_a = [m.get("chunk_id") for m in res_a["docs_metadados"][:k]]
            ids_b = [m.get("chunk_id") for m in res_b["docs_metadados"][:k]]
            mudou = "SIM" if ids_a != ids_b else "nao"

        pergunta_curta = item["pergunta"][:55] + "..." if len(item["pergunta"]) > 55 else item["pergunta"]
        linhas.append(f"| {id_} | {pergunta_curta} | {mod} | {hit_a} | {hit_b} | {mudou} |")

    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  AVALIACAO QUANTITATIVA DE RERANKING")
    print("  RRF (baseline) vs RRF + Cross-Encoder")
    print("=" * 70)

    # Filtra apenas perguntas REGULATORIO (com modulo_esperado definido)
    perguntas_regulatorio = [
        item for item in EVAL_SET_V2
        if item.get("categoria_esperada") == "REGULATORIO"
        and item.get("modulo_esperado") is not None
    ]
    print(f"\nDataset: {len(perguntas_regulatorio)} perguntas REGULATORIO com ground truth de modulo.")

    # -------------------------------------------------------------------------
    # Inicializa os dois retrievers
    # -------------------------------------------------------------------------
    print("\n[1/2] Inicializando retriever SEM reranker (baseline)...")
    t0 = time.time()
    retriever_sem = configurar_buscador(
        k_resultados=K_FINAL,
        k_candidatos=K_CANDIDATOS,
        k_rrf=K_RRF,
        use_reranker=False,
    )
    print(f"      Pronto em {time.time() - t0:.1f}s")

    print("\n[2/2] Inicializando retriever COM Cross-Encoder...")
    t0 = time.time()
    retriever_com = configurar_buscador(
        k_resultados=K_FINAL,
        k_candidatos=K_CANDIDATOS,
        k_rrf=K_RRF,
        use_reranker=True,
    )
    print(f"      Pronto em {time.time() - t0:.1f}s")

    # -------------------------------------------------------------------------
    # Avaliação
    # -------------------------------------------------------------------------
    resultados_a, metricas_a = avaliar_retriever("RRF (baseline)", retriever_sem, perguntas_regulatorio)
    resultados_b, metricas_b = avaliar_retriever("RRF + Cross-Encoder", retriever_com, perguntas_regulatorio)

    metricas_por_retriever = {
        "RRF (baseline)": metricas_a,
        "RRF + Cross-Encoder": metricas_b,
    }
    resultados_por_retriever = {
        "RRF (baseline)": resultados_a,
        "RRF + Cross-Encoder": resultados_b,
    }

    # -------------------------------------------------------------------------
    # Exibe tabela no terminal
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RESULTADOS")
    print("=" * 70)
    print()
    print(formatar_tabela_metricas(metricas_por_retriever))

    # Delta entre os dois retrievers
    print("\n--- Delta (Cross-Encoder - RRF baseline) ---")
    for k in KS:
        k_str = str(k)
        delta_hit  = metricas_b[k_str][f"hit_rate@{k}"]  - metricas_a[k_str][f"hit_rate@{k}"]
        delta_cp   = metricas_b[k_str][f"context_precision@{k}"] - metricas_a[k_str][f"context_precision@{k}"]
        delta_mrr  = metricas_b[k_str][f"mrr@{k}"]  - metricas_a[k_str][f"mrr@{k}"]
        delta_ndcg = metricas_b[k_str][f"ndcg@{k}"] - metricas_a[k_str][f"ndcg@{k}"]
        sinal = lambda v: f"+{v:.3f}" if v >= 0 else f"{v:.3f}"
        print(
            f"  K={k}: Hit {sinal(delta_hit)} | CP {sinal(delta_cp)} | "
            f"MRR {sinal(delta_mrr)} | NDCG {sinal(delta_ndcg)}"
        )

    # -------------------------------------------------------------------------
    # Salva relatórios
    # -------------------------------------------------------------------------
    relatorio_json = {
        "config": {
            "k_resultados": K_FINAL,
            "k_candidatos": K_CANDIDATOS,
            "k_rrf": K_RRF,
            "ks_avaliados": KS,
            "n_perguntas": len(perguntas_regulatorio),
        },
        "metricas": {
            retriever: {k: v for k, v in por_k.items()}
            for retriever, por_k in metricas_por_retriever.items()
        },
    }

    json_path = REPORTS_DIR / "reranker_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(relatorio_json, f, ensure_ascii=False, indent=2)

    md_path = REPORTS_DIR / "reranker_comparison.md"
    md_content = gerar_markdown(metricas_por_retriever, resultados_por_retriever, perguntas_regulatorio)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nRelatorio JSON salvo em: {json_path}")
    print(f"Relatorio MD  salvo em: {md_path}")
    print("\nAvaliacao concluida.\n")
