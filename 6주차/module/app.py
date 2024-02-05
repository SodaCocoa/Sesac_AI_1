# app.py
import streamlit as st

from streamlit_option_menu import option_menu
import image_classifier
import chatbot
import test  

PAGES = {
    "이미지 판별": image_classifier,
    "챗봇": chatbot,
    "카카오지도":test
}

st.sidebar.title('기능')
selection = st.sidebar.radio("페이지 선택", list(PAGES.keys()))

if selection in PAGES:

    page = PAGES[selection]
    page.app()

if "next_page" in st.session_state:
    if st.session_state.next_page == "챗봇":
        chatbot.app()
    elif st.session_state.next_page =="카카오지도":
        test.app()
