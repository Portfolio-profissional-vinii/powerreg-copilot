"""
src/agents/llm_factory.py

Fábrica centralizada de LLM para o Utilities Copilot.
Evita dependências circulares entre agentes e orquestradores.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _build_llm(temperature: float = 0):
    """
    Instancia o LLM com fallback automático usando frameworks LangChain.

    Prioridade:
        1. Groq (openai/gpt-oss-20b)
        2. Gemini 2.0 Flash (fallback)

    Args:
        temperature: Temperatura do modelo (0 para determinístico, >0 para geração).

    Returns:
        ChatGroq, ChatGoogleGenerativeAI ou modelo composto via with_fallbacks.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    models = []
    if groq_key:
        from langchain_groq import ChatGroq
        models.append(ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=temperature,
            groq_api_key=groq_key,
            max_retries=2,
            timeout=30,
        ))

    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        models.append(ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=temperature,
            google_api_key=google_key,
            max_retries=2,
        ))

    if not models:
        raise EnvironmentError(
            "Nenhuma API key encontrada. Configure GROQ_API_KEY ou GOOGLE_API_KEY no arquivo .env"
        )

    if len(models) == 1:
        return models[0]

    return models[0].with_fallbacks(models[1:])
