"""
app.py — Chat minimalista com IA usando Streamlit + OpenAI (carrega credenciais via JSON)

Requisitos de ambiente:
- Python 3.10+
- pip install streamlit openai python-dotenv

Como executar:
  streamlit run app.py
"""

import os
import json
import base64
from typing import List, Dict, Optional, Tuple

import streamlit as st
from openai import OpenAI

# -------------------------
# Configuração inicial UI
# -------------------------
st.set_page_config(page_title="Chat IA (OpenAI + Streamlit)", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
        /* Altura maior no chat_input */
        .stChatInput textarea {min-height: 3em;}
        /* Espaçamento agradável nos balões */
        .stChatMessage {padding-top: .25rem; padding-bottom: .25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Funções utilitárias (config)
# -------------------------

def parse_json_str(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def load_config_from_json(
    uploaded_file, pasted_json: str, file_path: str
) -> Tuple[Optional[dict], list[str]]:
    """Tenta carregar um dicionário de configuração de três fontes, nesta ordem:
    1) Arquivo enviado (uploader)
    2) Texto colado (pasted_json)
    3) Caminho local (file_path)
    Retorna (config, logs).
    """
    logs: list[str] = []

    # 1) upload
    if uploaded_file is not None:
        try:
            cfg = json.load(uploaded_file)
            logs.append("Config carregada do arquivo enviado.")
            return cfg, logs
        except Exception as e:
            logs.append(f"Falha ao ler JSON enviado: {e}")

    # 2) texto colado
    if pasted_json and pasted_json.strip():
        cfg = parse_json_str(pasted_json)
        if cfg is not None:
            logs.append("Config carregada do JSON colado.")
            return cfg, logs
        else:
            logs.append("JSON colado inválido.")

    # 3) caminho local
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            logs.append(f"Config carregada de: {file_path}")
            return cfg, logs
        except Exception as e:
            logs.append(f"Falha ao ler {file_path}: {e}")

    return None, logs


# -------------------------
# Sidebar: credenciais e opções
# -------------------------
with st.sidebar:
    st.header("🔐 Credenciais (via JSON)")
    st.caption(
        """O JSON deve conter chaves como `llm_api_key`, `llm_model` e opcionalmente `llm_base_url`.

Exemplo:
```json
{
  "llm_base_url": "https://api.openai.com/v1",
  "llm_model": "gpt-4o",
  "llm_api_key": "sk-..."
}
```"""
    )

    uploaded_cfg = st.file_uploader("Enviar arquivo JSON", type=["json"], key="cfg_uploader")
    pasted_cfg = st.text_area("Ou cole o JSON aqui", height=140)
    local_cfg_path = st.text_input("Ou aponte um caminho local (opcional)")

    cfg, cfg_logs = load_config_from_json(uploaded_cfg, pasted_cfg, local_cfg_path)

    if cfg_logs:
        for ln in cfg_logs:
            st.caption(f"• {ln}")

    # Fallbacks adicionais (opcionais)
    st.markdown("---")
    st.subheader("⚙️ Parâmetros do modelo")

    default_model = (cfg or {}).get("llm_model", "gpt-4o")
    model = st.selectbox(
        "Modelo",
        options=[
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
            "o3",
        ],
        index=(0 if default_model == "gpt-4.1" else 1 if default_model == "gpt-4o" else 2 if default_model == "gpt-4o-mini" else 3),
        help="Escolha o modelo conforme custo x qualidade.",
    )

    temperature = st.slider("Temperatura", 0.0, 1.5, 0.5, 0.1)

    system_prompt = st.text_area(
        "System prompt (opcional)",
        value=(
            "Você é um assistente útil, direto e técnico quando necessário."
        ),
        help="Define o comportamento geral do assistente.",
        height=120,
    )

# -------------------------
# Resolução da API key / base URL
# -------------------------
api_key = (cfg or {}).get("llm_api_key") or os.environ.get("OPENAI_API_KEY")
base_url = (cfg or {}).get("llm_base_url") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

if not api_key:
    st.warning("Forneça a chave via JSON (llm_api_key) ou defina OPENAI_API_KEY no ambiente.")
    st.stop()

# -------------------------
# Cliente OpenAI
# -------------------------
client = OpenAI(api_key=api_key, base_url=base_url)

# -------------------------
# Estado da conversa
# -------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render histórico
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"], caption=msg.get("caption", "imagem"))
        else:
            st.markdown(msg["content"])

# -------------------------
# Entrada do usuário (texto + upload opcional de imagem)
# -------------------------
col_input, col_upload = st.columns([0.75, 0.25])
with col_input:
    user_text = st.chat_input("Escreva sua mensagem…")
with col_upload:
    uploaded_image = st.file_uploader(
        "Imagem opcional", type=["png", "jpg", "jpeg"], label_visibility="collapsed", key="img_uploader"
    )

# -------------------------
# Monta input para Responses API (multimodal opcional)
# -------------------------
def make_input_items(history: List[Dict[str, str]], user_text: str, image_bytes: bytes | None):
    items = []
    if system_prompt:
        items.append({"role": "system", "content": system_prompt})

    # compacto: histórico em texto
    if history:
        history_text = []
        for m in history:
            role = "Usuário" if m.get("role") == "user" else "Assistente"
            if m.get("type") == "image":
                history_text.append(f"{role}: [imagem enviada]")
            else:
                history_text.append(f"{role}: {str(m.get('content', ''))}")
        items.append({"role": "user", "content": "\n".join(history_text)})

    if user_text:
        items.append({"role": "user", "content": user_text})

    if image_bytes is not None:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"
        items.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Analise a imagem enviada junto desta mensagem."},
                {"type": "input_image", "image_url": data_url},
            ],
        })

    return items

# -------------------------
# Envio da mensagem
# -------------------------
if user_text or uploaded_image is not None:
    if user_text:
        st.session_state["messages"].append({"role": "user", "content": user_text})
    if uploaded_image is not None:
        st.session_state["messages"].append({"role": "user", "type": "image", "content": uploaded_image})

    if user_text:
        with st.chat_message("user"):
            st.markdown(user_text)
    if uploaded_image is not None:
        with st.chat_message("user"):
            st.image(uploaded_image, caption="imagem enviada")

    image_bytes = uploaded_image.read() if uploaded_image is not None else None
    input_items = make_input_items(
        st.session_state["messages"][:-1] if uploaded_image else st.session_state["messages"], user_text, image_bytes
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            resp = client.responses.create(
                model=model,
                input=input_items,
                temperature=temperature,
            )
            assistant_text = resp.output_text or "(sem texto)"
            placeholder.markdown(assistant_text)
            st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
        except Exception as e:
            placeholder.error(f"Falha ao chamar a API: {e}")


# -------------------------
# Rodapé
# -------------------------
st.divider()
st.caption(
    "Exemplo educacional. Use JSON para credenciais (llm_api_key, llm_model, llm_base_url) e **NUNCA** versione sua chave."
)
