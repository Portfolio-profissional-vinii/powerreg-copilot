import os
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


st.set_page_config(
    page_title="PowerReg Copilot",
    page_icon="⚡",
    layout="wide",
)

from src.app.dashboard_tab import render_dashboard_tab
from src.agents.graph import CopilotOrchestrator

st.sidebar.title("🔑 Autenticação")

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""


user_api_key = st.sidebar.text_input(
    "Sua Groq API Key",
    type="password",
    value=st.session_state.groq_api_key,
    help="Obtenha sua chave em https://console.groq.com/keys",
)

if user_api_key:
    st.session_state.groq_api_key = user_api_key
    os.environ["GROQ_API_KEY"] = user_api_key

st.title("⚡ PowerReg Copilot")

tab_chat, tab_dash = st.tabs(
    [
        "💬 Chat Regulatório & Operacional",
        "📊 Dashboard de Indicadores",
    ]
)

with tab_chat:

    st.header("Assistente Virtual")

    # Inicializa histórico
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def renderizar_badge_categoria(categoria: str):
        if categoria == "REGULATORIO":
            st.caption("📘 **Agente Regulatório** (Normas ANEEL / PRODIST)")
        elif categoria == "OPERACIONAL":
            st.caption("📊 **Agente Operacional** (Dados Indicadores / DuckDB)")
        elif categoria == "SEM_RESPOSTA":
            st.caption("ℹ️ **Classificação:** Fora do escopo do sistema")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("categoria"):
                renderizar_badge_categoria(msg["categoria"])
            st.markdown(msg["content"])

    if not st.session_state.groq_api_key:

        st.info(
            "Insira sua **Groq API Key** na barra lateral "
            "para utilizar o Copiloto."
        )

    else:
        os.environ["GROQ_API_KEY"] = (
            st.session_state.groq_api_key
        )

        @st.cache_resource
        def carregar_orquestrador(api_key: str):
            os.environ["GROQ_API_KEY"] = api_key
            return CopilotOrchestrator()

        try:

            orchestrator = carregar_orquestrador(
                st.session_state.groq_api_key
            )

        except Exception as e:

            st.error(
                f"Não foi possível inicializar o Copiloto: {e}"
            )

            orchestrator = None

        if orchestrator is not None:

            prompt = st.chat_input(
                "Pergunte sobre regulamentação ou dados operacionais..."
            )

            if prompt:

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": prompt,
                    }
                )

                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):

                    with st.spinner(
                        "Analisando solicitação..."
                    ):

                        try:

                            detalhes = orchestrator.responder_com_detalhes(
                                prompt
                            )
                            categoria = detalhes.get("categoria", "REGULATORIO")
                            resposta = detalhes.get("resposta", "")

                            renderizar_badge_categoria(categoria)
                            st.markdown(resposta)

                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": resposta,
                                    "categoria": categoria,
                                }
                            )


                        except Exception as e:

                            st.error(
                                f"Erro ao processar a solicitação: {e}"
                            )

with tab_dash:

    render_dashboard_tab()