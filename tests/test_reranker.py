import pytest
from unittest.mock import MagicMock

from src.vectorstore.hybrid_search import HybridRetriever

class MockCrossEncoder:
    def __init__(self, model_name=None, device=None):
        pass

    def predict(self, pairs):
        """
        Mock do predict do CrossEncoder.
        pairs é uma lista de listas: [[query, texto], [query, texto]]
        """
        scores = []
        for q, text in pairs:
            # Lógica simples para mockar pontuações baseadas em palavras-chave para os testes de caso sensível
            text_lower = text.lower()
            has_nao_text = "não é permitido" in text_lower
            has_nao_q = "não" in q.lower()
            
            if has_nao_text and has_nao_q:
                scores.append(0.9)
            elif not has_nao_text and "é permitido" in text_lower and not has_nao_q:
                scores.append(0.9)
            elif not has_nao_text and "é permitido" in text_lower and has_nao_q:
                scores.append(0.1)
            elif has_nao_text and not has_nao_q:
                scores.append(0.1)
            else:
                # Pontuação baseada no tamanho do texto apenas para diferenciar
                scores.append(len(text) * 0.01)
        return scores


@pytest.fixture
def mock_retriever():
    # Criação de um mock para o HybridRetriever que foca no CrossEncoder
    retriever = HybridRetriever(
        client=MagicMock(),
        embeddings_model=MagicMock(),
        bm25=MagicMock(),
        corpus_texts=[
            "Texto irrelevante",
            "O acesso é permitido para manutenção.",
            "O acesso não é permitido para manutenção.",
            "Outro texto aleatório",
        ],
        corpus_metadados=[{}, {}, {}, {}],
        k=2,
        k_candidatos=4,
        k_rrf=4,
        cross_encoder=MockCrossEncoder(),
        collection_name="test"
    )
    return retriever


def test_cross_encoder_ordering_and_k_limit(mock_retriever):
    """
    Testa se o cross-encoder reordena os IDs baseados nas pontuações do modelo mockado
    e se corta os resultados pelo limite de self.k.
    """
    query = "Regras de manutenção"
    # IDs retornados pela etapa anterior (RRF) em uma ordem ruim
    top_ids_rrf = [0, 3, 1] 
    
    # O mock vai pontuar:
    # id=0: "Texto irrelevante" -> len=17 -> score 0.17
    # id=3: "Outro texto aleatório" -> len=21 -> score 0.21
    # id=1: "O acesso é permitido para manutenção." -> len=37 -> score 0.37
    
    final_ids = mock_retriever._aplicar_cross_encoder(query, top_ids_rrf)
    
    # Ordem esperada após rerank (decrescente): id=1, id=3, id=0
    # Como mock_retriever.k = 2, deve retornar apenas [1, 3]
    assert len(final_ids) == 2
    assert final_ids == [1, 3]


def test_cross_encoder_sensitive_cases_affirmative(mock_retriever):
    """
    Avalia casos sensíveis onde uma palavra muda o sentido (afirmação).
    O cross-encoder deve ser capaz de distinguir a nuance do contexto.
    """
    query = "O acesso é permitido para manutenção?"
    
    # IDs retornados pelo RRF: 2 (não permitido) em primeiro, 1 (permitido) em segundo
    top_ids_rrf = [2, 1]
    
    final_ids = mock_retriever._aplicar_cross_encoder(query, top_ids_rrf)
    
    # De acordo com nosso mock, a query NÃO contém "não".
    # id=1 ("é permitido") ganha score 0.9.
    # id=2 ("não é permitido") ganha score 0.1.
    # A ordem deve ser reajustada para [1, 2]
    assert final_ids[0] == 1


def test_cross_encoder_sensitive_cases_negative(mock_retriever):
    """
    Avalia casos sensíveis onde uma palavra muda o sentido (negação).
    """
    query = "O acesso não é permitido para manutenção?"
    
    # IDs retornados pelo RRF: 1 (permitido) em primeiro, 2 (não permitido) em segundo
    top_ids_rrf = [1, 2]
    
    final_ids = mock_retriever._aplicar_cross_encoder(query, top_ids_rrf)
    
    # A query CONTÉM "não".
    # id=2 ("não é permitido") ganha score 0.9.
    # id=1 ("é permitido") ganha score 0.1.
    # A ordem deve ser reajustada para [2, 1]
    assert final_ids[0] == 2


def test_cross_encoder_empty_input(mock_retriever):
    """
    Testa se a função de cross-encoder lida corretamente com input vazio do RRF.
    """
    final_ids = mock_retriever._aplicar_cross_encoder("query", [])
    assert final_ids == []

def test_cross_encoder_disabled(mock_retriever):
    """
    Testa comportamento quando não há reranker.
    """
    mock_retriever.cross_encoder = None
    top_ids_rrf = [3, 1, 2]
    final_ids = mock_retriever._aplicar_cross_encoder("query", top_ids_rrf)
    # Deve retornar a mesma ordem, mas cortada em self.k = 2
    assert final_ids == [3, 1]
