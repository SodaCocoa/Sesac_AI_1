# app.py
import streamlit as st

from streamlit_option_menu import option_menu
import image_classifier
import chatbot
import test  
import tmp 
PAGES = {
    "이미지 판별": image_classifier,
    "챗봇": chatbot,
    "카카오지도":test,
    '요금감면':tmp
}

st.sidebar.title('기능')
selection = st.sidebar.radio("페이지 선택", ["카카오지도","이미지 판별"])

# if selection in PAGES:

#     page = PAGES[selection]
page=PAGES[selection]
page.app()

if "next_page" in st.session_state:
    if st.session_state.next_page == "챗봇":
        chatbot.app()
    elif st.session_state.next_page =="카카오지도":
        test.app()
    elif st.session_state.next_page =="요금감면":
        tmp.app()
