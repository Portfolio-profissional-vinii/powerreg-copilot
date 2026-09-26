"""
src/agents/query_expander.py

Agente Expansor de Queries especializado no setor elétrico brasileiro (ANEEL/PRODIST).

Responsabilidade:
    Receber uma pergunta em linguagem natural (muitas vezes leiga ou ambígua)
    e expandi-la em 3 a 4 sub-queries complementares e estruturadas:
    1. Tradução da linguagem leiga para termos normativos oficiais da ANEEL;
    2. Foco na regra/módulo geral do PRODIST ou Resoluções Normativas (ex: REN 1000/2021);
    3. Foco em prazos, penalidades, exceções, responsabilidades e compensações financeiras.

Utiliza Pydantic e LangChain LCEL para geração estruturada, com fallback
automático para a query original em caso de falha de conexão, timeout ou erro de parsing.
"""

import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class QueryExpansionResult(BaseModel):
    """
    Estrutura Pydantic com as sub-queries especializadas para o PRODIST/ANEEL.
    """
    termo_normativo: str = Field(
        description="Tradução da pergunta leiga para terminologia técnica e vocabulário normativo oficial da ANEEL/PRODIST."
    )
    modulo_prodist: str = Field(
        description="Consulta focada no módulo geral do PRODIST pertinente (Módulos 1 a 11) ou Resolução Normativa (ex: REN 1000/2021)."
    )
    prazos_penalidades_compensacoes: str = Field(
        description="Consulta focada em prazos de atendimento, penalidades, regras de ressarcimento, limites regulatórios e compensações financeiras."
    )
    contexto_adicional: Optional[str] = Field(
        default=None,
        description="Consulta complementar opcional cobrindo procedimentos operacionais, medição ou responsabilidades da distribuidora."
    )

    def to_list(self) -> List[str]:
        """Retorna as sub-queries em formato de lista de strings válidas."""
        itens = [
            self.termo_normativo,
            self.modulo_prodist,
            self.prazos_penalidades_compensacoes,
        ]
        if self.contexto_adicional:
            itens.append(self.contexto_adicional)
        return [q.strip() for q in itens if q and q.strip()]


_SISTEMA_EXPANSAO_PROMPT = """Você é um Engenheiro Especialista em Regulação do Setor Elétrico Brasileiro (ANEEL e PRODIST).
Sua missão é receber a dúvida do usuário e gerar sub-queries altamente eficazes para recuperação precisa de documentos em um sistema RAG híbrido (Dense + BM25).

O acervo regulatório disponível compreende:
- Resolução Normativa ANEEL nº 1.000/2021 (Direitos e deveres do consumidor, prazos de atendimento, religação, faturamento, contestação)
- PRODIST Módulo 1 (Introdução e Conceitos Gerais)
- PRODIST Módulo 2 (Planejamento da Expansão)
- PRODIST Módulo 3 (Acesso ao Sistema de Distribuição e Conexão)
- PRODIST Módulo 4 (Procedimentos Operativos)
- PRODIST Módulo 5 (Sistemas de Medição)
- PRODIST Módulo 6 (Informações Requeridas e Obrigações)
- PRODIST Módulo 7 (Gestão de Perdas de Energia)
- PRODIST Módulo 8 (Qualidade do Fornecimento: DEC, FEC, DIC, FIC, DMIC, DICRI, níveis de tensão, conformidade)
- PRODIST Módulo 9 (Ressarcimento de Danos Elétricos em Equipamentos)
- PRODIST Módulo 10 (SIG-R - Sistema de Informação Geográfica Regulatório)
- PRODIST Módulo 11 (Faturamento e Cobrança)

DIRETRIZES DE EXPANSÃO:
1. termo_normativo: Converta termos informais para a linguagem estrita da ANEEL (ex: 'luz apagou' -> 'interrupção de energia continuidade', 'queimou geladeira' -> 'ressarcimento de dano elétrico em equipamento').
2. modulo_prodist: Indique expressamente o módulo provável ou a REN 1000 com o tema principal.
3. prazos_penalidades_compensacoes: Formule uma query focada nos prazos regulatórios (ex: dias úteis, corridos), penalidades à distribuidora, valores de compensação ou hipóteses de excludente de responsabilidade.
4. contexto_adicional (opcional): Adicione termos complementares sobre procedimentos, laudos ou medição caso seja relevante.

{format_instructions}
"""


class QueryExpander:
    """
    Agente responsável por expandir perguntas de usuários em múltiplas consultas regulatórias.
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        if llm is None:
            from src.agents.llm_factory import _build_llm
            # Temperatura baixa (0.1) para geração criativa porém controlada e aderente à regulação
            self.llm = _build_llm(temperature=0.1)
        else:
            self.llm = llm

        self.parser = PydanticOutputParser(pydantic_object=QueryExpansionResult)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SISTEMA_EXPANSAO_PROMPT),
            ("user", "Pergunta do usuário: {pergunta}"),
        ]).partial(format_instructions=self.parser.get_format_instructions())

        self.chain = self.prompt | self.llm | self.parser

    def expandir_estruturado(self, query: str) -> QueryExpansionResult:
        """
        Executa a expansão e retorna o objeto Pydantic estruturado.
        Em caso de erro, realiza fallback preenchendo a query original.
        """
        if not query or not query.strip():
            return QueryExpansionResult(
                termo_normativo="",
                modulo_prodist="",
                prazos_penalidades_compensacoes="",
            )

        try:
            resultado: QueryExpansionResult = self.chain.invoke({"pergunta": query.strip()})
            return resultado
        except Exception as exc:
            logger.warning(
                "Falha ao expandir query '%s' via LLM: %s. Aplicando fallback para a query original.",
                query,
                exc,
            )
            # Fallback seguro: replica a query original nos campos principais
            return QueryExpansionResult(
                termo_normativo=query.strip(),
                modulo_prodist=query.strip(),
                prazos_penalidades_compensacoes=query.strip(),
            )

    def expandir(self, query: str) -> List[str]:
        """
        Retorna a lista de sub-queries expandidas e deduplicadas.
        Garante fallback seguro para [query] se ocorrer qualquer erro.
        """
        if not query or not query.strip():
            return []

        try:
            estruturado = self.expandir_estruturado(query)
            itens = estruturado.to_list()
            
            # Deduplica sub-queries preservando a ordem
            vistos = set()
            sub_queries = []
            for item in itens:
                chave = item.strip().lower()
                if chave and chave not in vistos:
                    sub_queries.append(item.strip())
                    vistos.add(chave)

            if not sub_queries:
                return [query.strip()]
            return sub_queries
        except Exception as exc:
            logger.warning("Erro inesperado em expandir(): %s. Retornando query original.", exc)
            return [query.strip()]

    def obter_todas_as_queries(self, query: str, incluir_original: bool = True) -> List[str]:
        """
        Retorna a lista de sub-queries para envio aos mecanismos de busca.
        Se incluir_original for True, inclui a query original e deduplica mantendo a ordem.

        Args:
            query: Pergunta original do usuário.
            incluir_original: Se True, garante que a query original faça parte do lote.

        Returns:
            Lista de strings com as consultas para busca concorrente.
        """
        query_limpa = query.strip()
        if not query_limpa:
            return []

        sub_queries = self.expandir(query_limpa)

        resultado = []
        vistos = set()

        if incluir_original:
            resultado.append(query_limpa)
            vistos.add(query_limpa.lower())

        for sq in sub_queries:
            sq_lower = sq.lower().strip()
            if sq_lower and sq_lower not in vistos:
                resultado.append(sq.strip())
                vistos.add(sq_lower)

        return resultado


if __name__ == "__main__":
    import json
    expander = QueryExpander()
    teste = "Minha geladeira queimou depois que faltou luz. A empresa tem que pagar?"
    print(f"Query original: {teste}\n")
    
    print("Expandindo via LLM...")
    res = expander.expandir_estruturado(teste)
    print("Resultado estruturado:")
    print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False))
    
    print("\nTodas as queries para busca híbrida:")
    todas = expander.obter_todas_as_queries(teste)
    for i, q in enumerate(todas, 1):
        print(f"  {i}. {q}")
