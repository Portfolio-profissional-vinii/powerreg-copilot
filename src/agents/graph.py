"""
src/agents/graph.py

Orquestrador central do Copilot Total Utiliti.

Pipeline ativo:
    rag_agent.py → hybrid_search.py → Qdrant local (data/vectorstore)

LLM: Groq openai/gpt-oss-20b (rápido e determinístico para roteamento)
Fallback: gemini-2.0-flash via GOOGLE_API_KEY (quando GROQ_API_KEY ausente)

Notas arquiteturais:
    - regulatory_agent.py foi descontinuado após ablation test (Etapa 5):
        Hit@10 Hybrid 85.3% > Hit@10 Dense 76.5%
    - _build_llm() centraliza a criação do LLM para evitar duplicação entre módulos.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Garante acesso à raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from src.agents.rag_agent import iniciar_copilot
from src.agents.operations_agent import consultar_dados_operacionais


# ---------------------------------------------------------------------------
# Fábrica centralizada de LLM
# ---------------------------------------------------------------------------

from src.agents.llm_factory import _build_llm


# ---------------------------------------------------------------------------
# Orquestrador principal
# ---------------------------------------------------------------------------

class CopilotOrchestrator:
    def __init__(self):
        # Modelo roteador com temperatura 0 para decisão determinística
        self.router_llm = _build_llm(temperature=0)

        # Agente Regulatório (Hybrid RAG — BM25 + Dense + RRF, Qdrant local)
        self.regulatory_agent = iniciar_copilot()

        # Prompt para classificação de intenção — três categorias:
        #   OPERACIONAL  : dados numéricos, estatísticas, apurações e médias históricas de distribuidoras
        #   REGULATORIO  : normas, PRODIST, resoluções ANEEL, conceitos, fórmulas, limites e regras de indicadores
        #   SEM_RESPOSTA : perguntas fora do domínio do setor elétrico
        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é o roteador central do sistema Copilot Total Utiliti especializado no setor elétrico brasileiro.
Dada a pergunta do usuário, classifique a intenção em exatamente uma das três categorias:

- OPERACIONAL: Perguntas que solicitam dados numéricos específicos, estatísticas, consultas a valores apurados, médias, rankings, comparações quantitativas ou histórico de indicadores (ex: DEC, FEC) apurados por distribuidoras de energia (ex: 'Qual o DEC da CEA em 2020?', 'Qual distribuidora teve maior FEC em 2022?', 'Média de DEC apurado no ano X'). Requer consulta a banco de dados tabular.

- REGULATORIO: Perguntas conceituais, teóricas ou normativas sobre regras, leis, resoluções da ANEEL (ex: REN 1000), PRODIST (Módulos 1 a 11), definições de termos técnicos ou indicadores (ex: 'O que é DEC?', 'Como é calculado o DIC?', 'Quais os limites regulatórios do DEC?'), prazos de atendimento, ressarcimento por danos elétricos, regras de medição e faturamento. Requer consulta a documentos e normas.

- SEM_RESPOSTA: Perguntas que não têm relação com o setor elétrico, normativas da ANEEL/PRODIST ou dados de distribuidoras de energia (ex: piadas, receitas, previsão do tempo, outros setores).

Responda APENAS com uma palavra: 'OPERACIONAL', 'REGULATORIO' ou 'SEM_RESPOSTA'."""),
            ("user", "{pergunta}")
        ])

        self.router_chain = self.router_prompt | self.router_llm | StrOutputParser()

    def responder_com_detalhes(self, pergunta: str) -> dict:
        """
        Roteia a pergunta para o agente adequado e retorna categoria e resposta.

        Fluxo:
            1. Router LLM classifica a intenção (OPERACIONAL | REGULATORIO | SEM_RESPOSTA)
            2. Agente correspondente processa e retorna a resposta

        Args:
            pergunta: Pergunta do usuário em linguagem natural.

        Returns:
            dict com 'categoria' ('OPERACIONAL', 'REGULATORIO' ou 'SEM_RESPOSTA') e 'resposta' (str).
        """
        raw_output = self.router_chain.invoke({"pergunta": pergunta}).strip().upper()

        if "SEM_RESPOSTA" in raw_output:
            categoria = "SEM_RESPOSTA"
            resposta = (
                "Esta pergunta está fora do escopo do sistema. "
                "Sou especialista apenas em normativas ANEEL/PRODIST e dados "
                "operacionais de distribuidoras de energia elétrica."
            )
        elif "OPERACIONAL" in raw_output and "REGULATORIO" not in raw_output:
            categoria = "OPERACIONAL"
            resposta = self.consultar_operacional(pergunta)
        else:
            # Default → REGULATORIO (mantém comportamento original para
            # saídas inesperadas do LLM que não se encaixem nos padrões acima)
            categoria = "REGULATORIO"
            resposta = self.consultar_regulatorio(pergunta)

        return {
            "categoria": categoria,
            "resposta": resposta,
        }

    def responder(self, pergunta: str) -> str:
        """
        Roteia a pergunta para o agente adequado. Mantém compatibilidade com chamadas simples.

        Args:
            pergunta: Pergunta do usuário em linguagem natural.

        Returns:
            Resposta textual do agente correspondente.
        """
        return self.responder_com_detalhes(pergunta)["resposta"]

    def consultar_operacional(self, pergunta: str) -> str:
        return consultar_dados_operacionais(pergunta)

    def consultar_regulatorio(self, pergunta: str) -> str:
        return self.regulatory_agent.invoke(pergunta)


if __name__ == "__main__":
    orchestrator = CopilotOrchestrator()

    print("--- Teste 1: Dúvida Operacional ---")
    p1 = "Qual foi o valor médio do DEC em 2020?"
    print(f"Pergunta: {p1}")
    print(f"Resposta:\n{orchestrator.responder(p1)}\n")

    print("--- Teste 2: Dúvida Regulatória ---")
    p2 = "O que dizem as regras sobre ressarcimento por danos elétricos?"
    print(f"Pergunta: {p2}")
    print(f"Resposta:\n{orchestrator.responder(p2)}\n")

    print("--- Teste 3: Fora do domínio ---")
    p3 = "Me diga uma piada."
    print(f"Pergunta: {p3}")
    print(f"Resposta:\n{orchestrator.responder(p3)}\n")