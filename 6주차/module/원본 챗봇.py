# 필요한 라이브러리를 임포트합니다.
from keras.models import load_model  # TensorFlow 백엔드를 사용하는 Keras 모델을 로드하기 위해 필요
from PIL import Image, ImageOps  # 이미지 처리를 위한 라이브러리, PIL 대신 pillow를 사용
import numpy as np  # 수치 계산을 위한 라이브러리
import streamlit as st  # 웹 앱을 구축하기 위한 라이브러리
from dotenv import load_dotenv  # 환경 변수를 로드하기 위한 라이브러리
import os  # 운영 체제와 상호작용하기 위한 라이브러리, 환경 변수 사용 시 필요
import openai  # OpenAI API를 사용하기 위한 라이브러리
from openai import OpenAI  # OpenAI 클래스를 직접 사용하기 위해 임포트

# API 키를 초기화합니다. 보안상의 이유로 실제 키 값을 코드에 직접 넣지 않습니다.
key = ' '

# Streamlit 앱의 메인 함수입니다.
def app():

    st.title("🌱💬  SESAC BOT")  # 앱 타이틀 설정

    # OpenAI 클라이언트를 초기화합니다. 이를 통해 OpenAI API와 상호작용합니다.
    client = OpenAI(api_key=key)

    # Streamlit 세션 상태를 사용하여 챗봇의 메시지 기록을 관리합니다.
    # 'messages'가 세션 상태에 없다면 초기화합니다.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 저장된 챗 메시지를 화면에 표시합니다.
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue  # 시스템 메시지는 무시합니다.
        with st.chat_message(message["role"]):  # 사용자 또는 챗봇의 역할에 따라 메시지를 구분하여 표시합니다.
            st.markdown(message["content"])

    # 사용자 입력을 받습니다. st.chat_input을 사용하여 채팅 인터페이스에서 입력을 받습니다.
    if prompt := st.chat_input("하고 싶은 말 입력"):
        # 사용자의 입력을 세션 상태에 저장합니다.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):  # 사용자 메시지로 입력을 표시합니다.
            st.markdown(prompt)
        with st.chat_message("assistant"):  # 챗봇의 응답을 준비합니다.
            message_placeholder = st.empty()  # 응답을 동적으로 업데이트하기 위한 임시 플레이스홀더
            full_response = ""  # 응답 내용을 저장할 변수
            # OpenAI API를 사용하여 챗봇의 응답을 생성합니다.
            # for delta in client.chat.completions.create(...)를 통해 실시간으로 응답을 받고 업데이트합니다.
            for delta in client.chat.completions.create(
                model="gpt-4-turbo-preview",  # 사용할 모델
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,  # 응답을 스트림(실시간으로) 받습니다.
            ):
                # 응답 내용을 조합하여 full_response를 업데이트합니다.
                full_response += delta.choices[0].delta.content if delta.choices[0].delta.content else ""
                message_placeholder.markdown(full_response + "▌")  # 응답을 화면에 표시하고, 입력 중임을 나타냅니다.

            message_placeholder.markdown(full_response)  # 최종 응답을 화면에 표시합니다.
        # 챗봇의 응답을 세션 상태에 저장합니다.
        st.session_state.messages.append({"role": "assistant", "content": full_response})
