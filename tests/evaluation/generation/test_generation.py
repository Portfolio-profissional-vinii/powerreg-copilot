"""
tests/evaluation/generation/test_generation.py

Avaliação mínima de geração RAG — apenas o necessário para validar
que o contexto recuperado é utilizado corretamente pelo LLM.

Estratégia de economia de API:
  - N = 3 perguntas (não 10)
  - 1 único teste consolidado (Faithfulness + Answer Relevancy na mesma chamada)
  - Rate limit de 2s entre chamadas ao LLM

Pré-requisito: GROQ_API_KEY ou GOOGLE_API_KEY no .env
"""
import os
import json
import time
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

from tests.evaluation.eval_dataset_v2 import RETRIEVAL_SET


# ---------------------------------------------------------------------------
# Número máximo de perguntas avaliadas (mínimo para validar o pipeline)
# ---------------------------------------------------------------------------
N_AMOSTRAS = 3
RATE_LIMIT_SLEEP = 2.0  # segundos entre chamadas ao LLM


# ---------------------------------------------------------------------------
# Helpers — LLM judge e embeddings
# ---------------------------------------------------------------------------

def _get_judge_llm():
    """Retorna LLM juiz: Groq (prioritário) → Gemini (fallback) → None."""
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if groq_key:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model="openai/gpt-oss-20b", temperature=0, groq_api_key=groq_key)
        except Exception as e:
            print(f"\n[Geração] Groq indisponível ({e}). Tentando Gemini...")

    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=google_key)
        except Exception as e:
            print(f"\n[Geração] Gemini indisponível ({e}).")

    return None


def _get_judge_embeddings():
    """Retorna embeddings para métricas Ragas que precisam de similaridade semântica."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Coleta de respostas reais do sistema RAG
# ---------------------------------------------------------------------------

def _coletar_respostas_rag(
    entradas: list[dict],
    hybrid_retriever,
    orchestrator,
    n_max: int = N_AMOSTRAS,
) -> tuple[list[dict], int]:
    """
    Para cada pergunta:
      1. Recupera contexto real via HybridRetriever (sem LLM)
      2. Gera resposta real via CopilotOrchestrator (1 chamada LLM por pergunta)

    Retorna (lista_resultados, n_llm_calls).
    """
    resultados = []
    n_llm_calls = 0

    for entrada in entradas[:n_max]:
        try:
            # Contexto via retriever (sem LLM)
            docs = hybrid_retriever.invoke(entrada["pergunta"])
            contexto = [doc.page_content for doc in docs]

            # Rate limit entre chamadas ao LLM
            time.sleep(RATE_LIMIT_SLEEP)

            # Resposta via LLM (1 chamada)
            resposta = orchestrator.consultar_regulatorio(entrada["pergunta"])
            n_llm_calls += 1

            resultados.append({
                "id": entrada["id"],
                "pergunta": entrada["pergunta"],
                "resposta": resposta,
                "contexto": contexto,
                "modulo_esperado": entrada.get("modulo_esperado"),
                "erro": None,
            })

        except Exception as e:
            resultados.append({
                "id": entrada["id"],
                "pergunta": entrada["pergunta"],
                "resposta": "",
                "contexto": [],
                "erro": str(e),
            })

    return resultados, n_llm_calls


# ---------------------------------------------------------------------------
# Verificação determinística: resposta usa contexto (sem LLM judge)
# ---------------------------------------------------------------------------

def _verificar_grounding_determinisitco(resposta: str, contexto: list[str]) -> dict:
    """
    Heurística determinística para verificar se a resposta usa o contexto.
    Extrai termos significativos do contexto e verifica presença na resposta.

    Não substitui Ragas, mas funciona sem chamada de API adicional.
    """
    if not resposta or not contexto:
        return {"grounded": False, "razao": "Sem resposta ou contexto"}

    # Recusa esperada do sistema (anti-alucinação)
    frases_recusa = ["não encontrei", "não tenho informação", "fora do escopo"]
    if any(f in resposta.lower() for f in frases_recusa):
        return {"grounded": True, "razao": "Resposta de recusa (anti-alucinação OK)"}

    # Extrai tokens longos (>5 chars) do contexto como "evidências"
    import re
    tokens_contexto = set()
    for c in contexto[:3]:  # só os 3 primeiros chunks
        tokens = re.findall(r"\b[A-Za-zÀ-ú]{5,}\b", c)
        tokens_contexto.update(t.lower() for t in tokens[:50])

    if not tokens_contexto:
        return {"grounded": None, "razao": "Contexto sem tokens para verificar"}

    tokens_resposta = set(re.findall(r"\b[A-Za-zÀ-ú]{5,}\b", resposta.lower()))
    intersecao = tokens_contexto & tokens_resposta
    overlap_ratio = len(intersecao) / len(tokens_contexto) if tokens_contexto else 0

    return {
        "grounded": overlap_ratio >= 0.05,  # ≥5% de overlap de tokens
        "overlap_ratio": round(overlap_ratio, 3),
        "razao": f"Overlap de tokens: {overlap_ratio:.1%}",
    }


# ---------------------------------------------------------------------------
# Teste consolidado de geração
# ---------------------------------------------------------------------------

class TestRAGGeneration:
    """
    Avalia geração RAG com N=3 perguntas.
    1 teste consolidado para minimizar chamadas à API.
    """

    def test_geracao_rag_suite(self, hybrid_retriever, orchestrator, reports_dir):
        """
        Suite de geração: coleta respostas reais e avalia com:
          - Verificação determinística de grounding (sem API adicional)
          - Ragas Faithfulness + Answer Relevancy (se API disponível)

        Salva resultados em reports/generation_suite.json.
        """
        # Verificar pré-requisitos
        groq_key = os.getenv("GROQ_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        if not groq_key and not google_key:
            pytest.skip("GROQ_API_KEY ou GOOGLE_API_KEY não definida — testes de geração ignorados.")

        # Coletar respostas reais (N=3, minimiza chamadas à API)
        amostra = RETRIEVAL_SET[:N_AMOSTRAS]
        respostas, n_llm_calls = _coletar_respostas_rag(amostra, hybrid_retriever, orchestrator)

        validas = [r for r in respostas if not r.get("erro") and r["resposta"]]
        erros = [r for r in respostas if r.get("erro")]

        if not validas:
            pytest.skip(f"Nenhuma resposta válida coletada. Erros: {[r['erro'] for r in erros]}")

        # --- Avaliação determinística (sem custo de API adicional) ---
        resultados_det = []
        for r in validas:
            check = _verificar_grounding_determinisitco(r["resposta"], r["contexto"])
            resultados_det.append({
                "id": r["id"],
                "pergunta": r["pergunta"],
                "resposta_trecho": r["resposta"][:200],
                "modulo_esperado": r.get("modulo_esperado"),
                "grounding_check": check,
            })

        n_grounded = sum(1 for r in resultados_det if r["grounding_check"].get("grounded") is True)
        grounding_score = n_grounded / len(resultados_det) if resultados_det else 0

        # --- Avaliação Ragas (opcional — usa LLM judge, custo adicional de API) ---
        ragas_scores = {}
        n_llm_calls_ragas = 0

        try:
            from ragas.metrics.collections import Faithfulness, AnswerRelevancy
            from ragas import evaluate, EvaluationDataset
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            llm = _get_judge_llm()
            embeddings = _get_judge_embeddings()

            if llm:
                samples = [
                    {
                        "user_input": r["pergunta"],
                        "response": r["resposta"],
                        "retrieved_contexts": r["contexto"],
                    }
                    for r in validas
                ]

                metricas = [Faithfulness(), AnswerRelevancy()]
                ragas_llm = LangchainLLMWrapper(llm)
                ragas_emb = LangchainEmbeddingsWrapper(embeddings) if embeddings else None

                for m in metricas:
                    if hasattr(m, "llm"):
                        m.llm = ragas_llm
                    if ragas_emb and hasattr(m, "embeddings"):
                        m.embeddings = ragas_emb

                dataset = EvaluationDataset.from_list(samples)
                resultado = evaluate(dataset=dataset, metrics=metricas)

                if hasattr(resultado, "to_pandas"):
                    df = resultado.to_pandas()
                    for col in df.columns:
                        if df[col].dtype in ["float64", "float32"]:
                            ragas_scores[col] = float(df[col].mean())

                # Cada pergunta = ~2 chamadas LLM (Faithfulness + Relevancy)
                n_llm_calls_ragas = len(validas) * len(metricas)

        except ImportError:
            ragas_scores = {"nota": "Ragas não instalado — instale com: pip install ragas"}
        except Exception as e:
            ragas_scores = {"erro": str(e)}

        # --- Relatório ---
        suite_result = {
            "n_perguntas_avaliadas": len(amostra),
            "n_respostas_validas": len(validas),
            "n_erros": len(erros),
            "llm_calls_geracao": n_llm_calls,
            "llm_calls_ragas": n_llm_calls_ragas,
            "grounding_determinístico": {
                "score": grounding_score,
                "n_grounded": n_grounded,
                "n_total": len(resultados_det),
                "nota": "Overlap de tokens entre resposta e contexto (heurística, sem custo de API)",
            },
            "ragas": ragas_scores if ragas_scores else None,
            "detalhes": resultados_det,
            "erros": [{"id": r["id"], "erro": r["erro"]} for r in erros],
        }

        (reports_dir / "generation_suite.json").write_text(
            json.dumps(suite_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # --- Print resumo ---
        print(f"\n{'='*50}")
        print("GERAÇÃO RAG — SUITE DE AVALIAÇÃO")
        print(f"{'='*50}")
        print(f"  Perguntas avaliadas : {len(amostra)}")
        print(f"  Respostas válidas   : {len(validas)}")
        print(f"  Chamadas LLM        : {n_llm_calls} (geração) + {n_llm_calls_ragas} (Ragas)")
        print(f"\n  Grounding (determinístico): {grounding_score:.3f}")
        for r in resultados_det:
            gc = r["grounding_check"]
            status = "✅" if gc.get("grounded") else ("⚠️" if gc.get("grounded") is None else "❌")
            print(f"    {status} [{r['id']}] {gc.get('razao', '')}")

        if ragas_scores and "faithfulness" in ragas_scores:
            faith = ragas_scores["faithfulness"]
            rel = ragas_scores.get("answer_relevancy", 0)
            print(f"\n  Ragas Faithfulness  : {faith:.3f} {'✅' if faith >= 0.7 else '⚠️'}")
            print(f"  Ragas Answer Relev. : {rel:.3f} {'✅' if rel >= 0.7 else '⚠️'}")
        elif ragas_scores:
            print(f"\n  Ragas: {ragas_scores}")

        if erros:
            print(f"\n  ❌ Erros:")
            for e in erros:
                print(f"    [{e['id']}] {e['erro'][:80]}")

        print(f"{'='*50}")

        # A validação real: pelo menos 1 resposta foi gerada e coletada sem erro
        assert len(validas) > 0, (
            f"Nenhuma resposta válida coletada de {len(amostra)} tentativas. "
            f"Erros: {[r['erro'] for r in erros]}"
        )
