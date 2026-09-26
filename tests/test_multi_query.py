"""
tests/test_multi_query.py

Testes unitários e de integração para a Arquitetura Multi-Query Retrieval:
1. Reciprocal Rank Fusion (RRF) & Deduplicação
2. Query Expander Agent & Fallback Seguro
3. Execução Concorrente e MultiQueryHybridRetriever
"""

import time
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.utils.rrf import calcular_score_rrf, reciprocal_rank_fusion
from src.agents.query_expander import QueryExpander, QueryExpansionResult
from src.retrieval.multi_query import MultiQueryHybridRetriever


# =============================================================================
# 1. TESTES DO MÓDULO RRF & DEDUPLICAÇÃO
# =============================================================================

def test_calcular_score_rrf():
    """Valida a fórmula 1 / (k + rank) com k padrão e customizado."""
    # k = 60 (padrão)
    assert calcular_score_rrf(1, k=60) == pytest.approx(1.0 / 61.0)
    assert calcular_score_rrf(2, k=60) == pytest.approx(1.0 / 62.0)
    
    # k customizado
    assert calcular_score_rrf(1, k=20) == pytest.approx(1.0 / 21.0)
    assert calcular_score_rrf(5, k=10) == pytest.approx(1.0 / 15.0)

    with pytest.raises(ValueError):
        calcular_score_rrf(0)


def test_reciprocal_rank_fusion_basico():
    """Valida a fusão simples de duas listas sem duplicatas."""
    doc1 = Document(page_content="Artigo 1", metadata={"chunk_id": 1, "modulo": "Módulo 8"})
    doc2 = Document(page_content="Artigo 2", metadata={"chunk_id": 2, "modulo": "Módulo 8"})
    doc3 = Document(page_content="Artigo 3", metadata={"chunk_id": 3, "modulo": "Módulo 9"})

    lista_a = [doc1, doc2]
    lista_b = [doc3]

    resultado = reciprocal_rank_fusion([lista_a, lista_b], k=60, top_n=3)

    assert len(resultado) == 3
    # doc1 e doc3 foram 1º colocados nas suas respectivas listas (rank 1 -> 1/61)
    # doc2 foi 2º colocado na lista A (rank 2 -> 1/62)
    assert resultado[0].metadata["chunk_id"] in [1, 3]
    assert resultado[1].metadata["chunk_id"] in [1, 3]
    assert resultado[2].metadata["chunk_id"] == 2
    assert resultado[2].metadata["rrf_score"] == pytest.approx(1.0 / 62.0)


def test_reciprocal_rank_fusion_deduplicacao_e_metadados():
    """Valida que documentos presentes em múltiplas listas têm seus scores acumulados."""
    doc_comum = Document(page_content="Regra geral de ressarcimento", metadata={"chunk_id": 42, "modulo": "Módulo 9", "pagina": 15})
    doc_outro = Document(page_content="Regra de DEC/FEC", metadata={"chunk_id": 99, "modulo": "Módulo 8", "pagina": 30})

    # doc_comum aparece em 1º na lista 1 e em 2º na lista 2
    lista_1 = [doc_comum, doc_outro]
    lista_2 = [doc_outro, doc_comum]

    # k = 60
    # doc_comum score: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    # doc_outro score: 1/(60+2) + 1/(60+1) = 1/62 + 1/61
    resultado = reciprocal_rank_fusion([lista_1, lista_2], k=60, top_n=5)

    assert len(resultado) == 2  # Deduplicado de 4 para 2
    for doc in resultado:
        assert doc.metadata["rrf_occurrences"] == 2
        assert len(doc.metadata["rrf_ranks"]) == 2
        assert doc.metadata["rrf_score"] == pytest.approx((1.0 / 61.0) + (1.0 / 62.0))
        assert "modulo" in doc.metadata
        assert "pagina" in doc.metadata


def test_reciprocal_rank_fusion_limite_top_n():
    """Valida o truncamento para top_n."""
    docs = [Document(page_content=f"Doc {i}", metadata={"chunk_id": i}) for i in range(20)]
    resultado = reciprocal_rank_fusion([docs], k=60, top_n=5)
    assert len(resultado) == 5
    assert resultado[0].metadata["chunk_id"] == 0


def test_reciprocal_rank_fusion_listas_vazias():
    """Valida comportamento resiliente para listas vazias."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# =============================================================================
# 2. TESTES DO QUERY EXPANDER AGENT
# =============================================================================

def test_query_expander_modelo_pydantic():
    """Valida a estrutura do schema Pydantic de expansão."""
    ex = QueryExpansionResult(
        termo_normativo="interrupção fornecimento energia limites continuidade",
        modulo_prodist="PRODIST Módulo 8 Qualidade do Fornecimento",
        prazos_penalidades_compensacoes="compensação financeira descumprimento DIC FIC",
        contexto_adicional="apuração mensal e faturamento",
    )
    lista = ex.to_list()
    assert len(lista) == 4
    assert "interrupção fornecimento" in lista[0]
    assert "Módulo 8" in lista[1]


def test_query_expander_fallback_em_erro():
    """Valida que em caso de falha do LLM, o fallback para a query original é executado."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("Simulação de timeout ou API indisponível")

    expander = QueryExpander(llm=mock_llm)
    
    pergunta_teste = "Qual o prazo para ressarcimento?"
    resultado = expander.expandir(pergunta_teste)
    
    # Fallback seguro: retorna uma lista contendo a query original
    assert len(resultado) == 1
    assert resultado[0] == pergunta_teste

    # Valida método de todas as queries
    todas = expander.obter_todas_as_queries(pergunta_teste, incluir_original=True)
    assert todas == [pergunta_teste]


def test_query_expander_com_mock_sucesso():
    """Valida expansão quando o LLM responde conforme esperado."""
    expander = QueryExpander.__new__(QueryExpander)
    expander.chain = MagicMock()
    expander.chain.invoke.return_value = QueryExpansionResult(
        termo_normativo="ressarcimento dano eletrico equipamento",
        modulo_prodist="PRODIST Módulo 9 e REN 1000",
        prazos_penalidades_compensacoes="prazo 15 dias uteis deferimento",
    )

    pergunta = "Queimou minha geladeira, a distribuidora paga?"
    sub_queries = expander.expandir(pergunta)
    assert len(sub_queries) == 3
    assert "ressarcimento dano" in sub_queries[0]

    todas = expander.obter_todas_as_queries(pergunta, incluir_original=True)
    assert len(todas) == 4
    assert todas[0] == pergunta


# =============================================================================
# 3. TESTES DE CONCORRÊNCIA E MULTI-QUERY RETRIEVER
# =============================================================================

def test_multi_query_retriever_execucao_concorrente():
    """
    Valida a execução paralela das sub-queries no MultiQueryHybridRetriever.
    Verifica que 4 queries com atraso de 0.1s rodam em paralelo (tempo total << 0.4s).
    """
    mock_expander = MagicMock()
    mock_expander.obter_todas_as_queries.return_value = [
        "query_0_original",
        "query_1_normativa",
        "query_2_modulo",
        "query_3_prazos",
    ]

    mock_hybrid = MagicMock()
    def fake_invoke(query_str):
        time.sleep(0.1)  # Simula latência de rede/busca
        return [
            Document(page_content=f"Resultado para {query_str}", metadata={"chunk_id": hash(query_str) % 1000})
        ]
    mock_hybrid.invoke.side_effect = fake_invoke

    retriever = MultiQueryHybridRetriever(
        hybrid_retriever=mock_hybrid,
        query_expander=mock_expander,
        k=5,
        rrf_k=60,
        max_workers=4,
        incluir_original=True,
    )

    inicio = time.perf_counter()
    docs = retriever.invoke("Minha luz caiu várias vezes esse mês")
    duracao = time.perf_counter() - inicio

    assert len(docs) == 4
    # Se fosse sequencial demoraria >= 0.40s. Em paralelo deve levar ~0.15s (menos de 0.35s).
    assert duracao < 0.35, f"Execução não foi paralela! Demorou {duracao:.3f}s"
    
    # Verifica que todos os documentos possuem metadados de RRF
    for doc in docs:
        assert "rrf_score" in doc.metadata
        assert "rrf_occurrences" in doc.metadata


@pytest.mark.asyncio
async def test_multi_query_retriever_execucao_assincrona():
    """Valida a execução assíncrona com ainvoke."""
    mock_expander = MagicMock()
    mock_expander.obter_todas_as_queries.return_value = ["q1", "q2"]

    mock_hybrid = MagicMock()
    mock_hybrid.invoke.side_effect = lambda q: [
        Document(page_content=f"Doc {q}", metadata={"chunk_id": hash(q) % 500})
    ]

    retriever = MultiQueryHybridRetriever(
        hybrid_retriever=mock_hybrid,
        query_expander=mock_expander,
        k=5,
        rrf_k=60,
    )

    docs = await retriever.ainvoke("Pergunta teste async")
    assert len(docs) == 2
    assert all("rrf_score" in d.metadata for d in docs)


def test_reciprocal_rank_fusion_fallback_hash_sem_chunk_id():
    """Valida deduplicação por hash de conteúdo quando chunk_id não está presente nos metadados."""
    doc_a1 = Document(page_content="Texto idêntico nas duas fontes", metadata={"modulo": "Módulo 8"})
    doc_a2 = Document(page_content="Texto idêntico nas duas fontes", metadata={"modulo": "Módulo 8"})

    resultado = reciprocal_rank_fusion([[doc_a1], [doc_a2]], k=60, top_n=5, id_key="chunk_id")
    assert len(resultado) == 1
    assert resultado[0].metadata["rrf_occurrences"] == 2

