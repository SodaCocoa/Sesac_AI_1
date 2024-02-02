# image_classifier.py
import streamlit as st
from PIL import Image
from keras.models import load_model
import numpy as np
from PIL import ImageOps  # Install pillow instead of PIL
import numpy as np
from dotenv import load_dotenv 
import os
import image_classifier
import chatbot
import cv2
import streamlit as st
import requests 

# Streamlit 애플리케이션 구성
def app():
    st.title("카카오지도")

    # # Kakao API 키
    api_key = "0398906c95ecf59778b8289319fd7553"  # 자신의 Kakao API 키로 대체해주세요.

    # Kakao 지도의 중심 좌표로 서울을 설정
    center_coordinates = (37.5665, 126.9780)  # 서울의 위도(latitude)와 경도(longitude)

    # Kakao 지도 웹 페이지의 URL
    kakao_map_url = f"https://map.kakao.com/link/search/공영주차장?urlLevel=8&lat={center_coordinates[0]}&lng={center_coordinates[1]}&apikey={api_key}"

    # Streamlit의 iframe을 사용하여 외부 웹 페이지 표시
    st.components.v1.iframe(kakao_map_url, width=800, height=600)
    # if st.button("가장 가까운 공영주차장 찾기"):
    #     # "공영주차장" 키워드로 Kakao 지도 API에 요청을 보냅니다.
    #     search_query = "공영주차장"
    #     search_url = f"https://dapi.kakao.com/v2/local/search/keyword.json?query={search_query}"
    #     headers = {"Authorization": f"KakaoAK {api_key}"}

    #     response = requests.get(search_url, headers=headers)
    #     data = response.json()

    #     if data.get("documents"):
    #         # 검색 결과 중에서 가장 첫 번째 결과를 선택합니다.
    #         first_result = data["documents"][0]

    #         # 선택된 결과의 좌표를 가져옵니다.
    #         lat, lng = first_result["y"], first_result["x"]

    #         # Kakao 지도 길찾기 URL을 생성합니다.
    #         kakao_map_url = f"https://map.kakao.com/link/to/{search_query},{lat},{lng}"

    #         # Streamlit의 iframe을 사용하여 길찾기 링크를 표시합니다.
    #         st.components.v1.iframe(kakao_map_url, width=800, height=600)
    #     else:
    #         st.error("가장 가까운 공영주차장을 찾을 수 없습니다.")
    