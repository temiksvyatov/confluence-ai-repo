import streamlit as st
import httpx
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="Confluence RAG Бот", layout="wide")
st.title("Помощник по базе знаний Confluence")

if "session_id" not in st.session_state:
    st.session_state.session_id = st.text_input("ID сессии", value="default").strip() or "default"
session_id = st.session_state.session_id

with st.expander("Переиндексация (админ)", expanded=False):
    pw = st.text_input("Пароль", type="password")
    if st.button("Запустить переиндексацию"):
        r = httpx.post(f"{BACKEND_URL}/reindex", json={"password": pw}, timeout=10)
        st.write("Запущено!" if r.status_code == 200 else f"Ошибка {r.status_code}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    uploaded = st.file_uploader("Прикрепить файлы", accept_multiple_files=True, key=f"f{len(st.session_state.messages)}")

    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            files_to_send = []
            if uploaded:
                for f in uploaded:
                    files_to_send.append(("files", (f.name, f.getvalue(), f.type)))

            payload = {"message": prompt, "session_id": session_id}

            try:
                with httpx.stream(
                    method="POST",
                    url=f"{BACKEND_URL}/chat",
                    data=payload,
                    files=files_to_send or None,
                    timeout=300.0
                ) as r:
                    r.raise_for_status()
                    text = ""
                    placeholder = st.empty()
                    for chunk in r.iter_text():
                        text += chunk
                        placeholder.markdown(text + "▌")
                    placeholder.markdown(text)

                st.session_state.messages.append({"role": "assistant", "content": text})

            except Exception as e:
                st.error(f"Ошибка: {e}")
