import streamlit as st

def render_chat_tab(orchestrator):
    st.subheader("💬 Chat Inteligente")
    st.caption("Consulte normativas do PRODIST ou dados operacionais da ANEEL em linguagem natural.")

    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    def renderizar_badge(categoria: str):
        if categoria == "REGULATORIO":
            st.caption("📘 **Agente Regulatório** (Normas ANEEL / PRODIST)")
        elif categoria == "OPERACIONAL":
            st.caption("📊 **Agente Operacional** (Dados Indicadores / DuckDB)")
        elif categoria == "SEM_RESPOSTA":
            st.caption("ℹ️ **Classificação:** Fora do escopo do sistema")

    # Exibe o histórico do chat
    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            if msg.get("categoria"):
                renderizar_badge(msg["categoria"])
            st.markdown(msg["content"])

    pergunta = st.chat_input("Digite sua dúvida sobre regras ou dados da ANEEL...")

    if pergunta:
        with st.chat_message("user"):
            st.markdown(pergunta)
        
        st.session_state.mensagens.append({"role": "user", "content": pergunta})
        
        with st.chat_message("assistant"):
            with st.spinner("Analisando e consultando bases de dados..."):
                try:
                    if hasattr(orchestrator, "responder_com_detalhes"):
                        detalhes = orchestrator.responder_com_detalhes(pergunta)
                        categoria = detalhes.get("categoria", "REGULATORIO")
                        resposta = detalhes.get("resposta", "")
                    else:
                        categoria = None
                        resposta = orchestrator.responder(pergunta)

                    if categoria:
                        renderizar_badge(categoria)
                    st.markdown(resposta)
                    st.session_state.mensagens.append({
                        "role": "assistant",
                        "content": resposta,
                        "categoria": categoria,
                    })
                except Exception as e:
                    st.error(f"Erro ao processar resposta: {e}")