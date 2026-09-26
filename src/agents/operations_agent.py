"""
src/agents/operations_agent.py

Agente Operacional — Tradução NL→SQL sobre dados de indicadores de distribuidoras.

LLM: Groq openai/gpt-oss-20b (temperatura 0 para geração SQL determinística)
Base: DuckDB lendo data/processed/aneel_indicadores.parquet
Dados: DEC, FEC e variantes por distribuidora (EAC, CEA, ETO, ...), 2020–2026

Fluxo:
    Pergunta → LLM gera SQL → DuckDB executa → LLM formula resposta em linguagem natural
"""

import os
import sys
import duckdb
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Carrega variáveis de ambiente do arquivo .env
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

PARQUET_PATH = ROOT_DIR / "data" / "processed" / "aneel_indicadores.parquet"


# ---------------------------------------------------------------------------
# Fábrica centralizada de LLM (Groq → Gemini fallback)
# ---------------------------------------------------------------------------

def _build_llm(temperature: float = 0):
    """
    Instancia o LLM com prioridade Groq openai/gpt-oss-20b → Gemini 2.0 Flash.

    Args:
        temperature: 0 para SQL determinístico, >0 para resposta em linguagem natural.

    Returns:
        ChatGroq ou ChatGoogleGenerativeAI.

    Raises:
        EnvironmentError: se nenhuma API key disponível.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=temperature,
            groq_api_key=groq_key,
            max_retries=3,
            timeout=60,
        )

    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=temperature,
            google_api_key=google_key,
            max_retries=3,
        )

    raise EnvironmentError(
        "Configure GROQ_API_KEY ou GOOGLE_API_KEY no arquivo .env"
    )


# ---------------------------------------------------------------------------
# Schema do prompt SQL — documentado explicitamente para o LLM
# ---------------------------------------------------------------------------

_SQL_SCHEMA = """Tabela disponível: `indicadores`
Colunas da tabela:
- SigAgente (TEXT): Sigla da distribuidora (ex: 'EAC', 'CEA', 'ETO', 'EQUATORIAL')
- SigIndicador (TEXT): Nome do indicador (ex: 'DEC', 'FEC', 'DECXNC', 'FECIPC')
- DscConjUndConsumidoras (TEXT): Nome do conjunto de unidades consumidoras
- AnoIndice (BIGINT): Ano de referência do indicador (ex: 2020)
- NumPeriodoIndice (BIGINT): Mês ou período do indicador (1 a 12)
- VlrIndiceEnviado (DOUBLE): Valor numérico apurado"""

_SQL_RULES = """Regras Importantes:
1. Retorne APENAS o código SQL puro (sem tags de código Markdown como ```sql).
2. Use ILIKE para buscas de texto flexíveis (ex: SigAgente ILIKE '%CEA%').
3. Selecione apenas as colunas relevantes ou agregados (SUM, AVG, COUNT, MAX, MIN).
4. Sempre use aliases descritivos em agregações (ex: AVG(VlrIndiceEnviado) AS media_dec).
5. Para filtros de texto exato em siglas, prefira: SigIndicador = 'DEC' em vez de ILIKE."""


# ---------------------------------------------------------------------------
# Execução SQL via DuckDB
# ---------------------------------------------------------------------------

def executar_query_sql(sql_query: str) -> str:
    """
    Executa consultas SQL via DuckDB no arquivo Parquet de indicadores.

    Args:
        sql_query: Instrução SQL válida para DuckDB referenciando a view `indicadores`.

    Returns:
        Resultado formatado como string tabular, ou mensagem de erro/vazio.
    """
    try:
        con = duckdb.connect()
        con.execute(f"CREATE VIEW indicadores AS SELECT * FROM '{PARQUET_PATH}'")
        df_resultado = con.execute(sql_query).df()
        con.close()

        if df_resultado.empty:
            return "Nenhum resultado encontrado para os filtros informados."
        return df_resultado.to_string(index=False)
    except Exception as e:
        return f"Erro ao executar consulta na base de dados: {e}"


# ---------------------------------------------------------------------------
# Pipeline completo: Pergunta → SQL → Resultado → Resposta Natural
# ---------------------------------------------------------------------------

def consultar_dados_operacionais(pergunta_usuario: str) -> str:
    """
    Traduz a pergunta do usuário em SQL, consulta a base e formula a resposta final.

    Componentes:
        1. LLM SQL (temperatura=0): NL → SQL DuckDB
        2. DuckDB: executa SQL sobre Parquet de indicadores
        3. LLM Resposta (temperatura=0.1): dados tabulares → linguagem natural

    Args:
        pergunta_usuario: Pergunta em linguagem natural sobre indicadores operacionais.

    Returns:
        Resposta em linguagem natural baseada nos dados da consulta SQL.
    """
    # --- Etapa 1: Gerar SQL ---
    llm_sql = _build_llm(temperature=0)

    prompt_sql = ChatPromptTemplate.from_messages([
        ("system", f"""Você é um especialista em SQL para DuckDB. Sua tarefa é converter a pergunta do usuário em uma instrução SQL válida.

{_SQL_SCHEMA}

{_SQL_RULES}"""),
        ("user", "{pergunta}")
    ])

    cadeia_sql = prompt_sql | llm_sql | StrOutputParser()
    query_gerada = cadeia_sql.invoke({"pergunta": pergunta_usuario}).strip()
    # Remove blocos de código Markdown residuais
    query_gerada = query_gerada.replace("```sql", "").replace("```", "").strip()

    # --- Etapa 2: Executar SQL ---
    resultado_dados = executar_query_sql(query_gerada)

    # --- Etapa 3: Formular resposta em linguagem natural ---
    llm_resposta = _build_llm(temperature=0.1)

    prompt_resposta = ChatPromptTemplate.from_messages([
        ("system", (
            "Você é o Copilot Operacional da Total Utiliti. "
            "Responda a dúvida do usuário de forma clara e direta com base EXCLUSIVAMENTE "
            "nos dados tabulares retornados pela consulta SQL. "
            "Não invente valores que não estejam nos dados. "
            "Se os dados estiverem vazios ou contiverem erro, informe isso claramente."
        )),
        ("user", "Pergunta do Usuário: {pergunta}\n\nDados Encontrados:\n{dados}\n\nResposta:")
    ])

    cadeia_resposta = prompt_resposta | llm_resposta | StrOutputParser()
    return cadeia_resposta.invoke({"pergunta": pergunta_usuario, "dados": resultado_dados})


if __name__ == "__main__":
    pergunta_teste = "Qual foi o valor médio do indicador DEC no ano de 2020?"
    print(f"❓ Pergunta: {pergunta_teste}\n")
    resposta = consultar_dados_operacionais(pergunta_teste)
    print(f"🤖 Copilot Operacional:\n{resposta}")