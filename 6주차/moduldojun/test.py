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
from time import sleep
import sys
import time
import folium
from streamlit_folium import st_folium
KAKAO_API_KEY="dbbe189cd18054f0ab68f6c59ac3592e"
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
def find_nearest_parking(lat, lng):
    url = f"https://dapi.kakao.com/v2/local/search/category.json?category_group_code=PK6&x={lng}&y={lat}&radius=2000"
    response = requests.get(url, headers=headers)
    return response.json()

def get_route(start_lat, start_lng, end_lat, end_lng):
    url = "https://apis-navi.kakaomobility.com/v1/directions"
    params = {
        'origin': f'{start_lng},{start_lat}',
        'destination': f'{end_lng},{end_lat}',
        'waypoints': '',
        'priority': 'RECOMMEND',
    }
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        route_data = response.json()
        # 여기에서는 간단히 첫 번째 경로의 좌표들만 반환합니다.
        # 실제로는 응답에서 더 많은 정보를 추출하고 처리할 필요가 있습니다.
        routes = route_data['routes']  # 모든 경로를 가져옴
        # print(routes)
        return routes
        # coords = route_data['routes'][0]['sections'][0]['roads'][0]['vertexes']
        # return coords
    else:
        return None
def show_routes_on_map(routes, start_lat, start_lng, end_lat, end_lng):
    if routes:
        # 출발지와 도착지 좌표를 튜플로 생성
        start_location = (start_lat, start_lng)
        end_location = (end_lat, end_lng)
        
        # 지도 생성
        m = folium.Map(location=start_location, zoom_start=13)
        
        for route in routes:
            roads_vertexes = []
            for section in route['sections']:
                for road in section['roads']:
                    if 'vertexes' in road:
                        roads_vertexes.extend(road['vertexes'])
            # print(roads_vertexes)
            # print('------'*20)
            m = folium.Map(location=[37.654484119125854, 127.04919345753622], zoom_start=15)  # 맵의 초기 위치 및 줌 레벨 설정

            # 경로를 표시하기 위한 PolyLine 추가
            path = folium.PolyLine(
                locations=[(roads_vertexes[i + 1], roads_vertexes[i]) for i in range(0, len(roads_vertexes), 2)],
                color='blue',  # 경로 색상 설정
                weight=5,  # 경로 두께 설정
                opacity=0.7,  # 투명도 설정
            ).add_to(m)
    #         # 경로 좌표 가져오기
            
    #         coords = route['sections'][0]['roads'][0]['vertexes']
    #         coords = [(coord['y'], coord['x']) for coord in coords]  # 좌표를 튜플 형태로 변환
            
    #         # 경로 그리기
    #         folium.PolyLine(locations=coords, color="blue", weight=2.5, opacity=1).add_to(m)
        
    #     # 출발지와 도착지에 마커 추가
        folium.Marker(start_location, icon=folium.Icon(color='green')).add_to(m)
        folium.Marker(end_location, icon=folium.Icon(color='red')).add_to(m)
        
        # Streamlit 앱에 지도 표시
        st_folium(m, width=725, height=500)
    else:
        st.write("경로 정보를 가져올 수 없습니다.")
        
# 구성
def app():
    st.title('가장 가까운 공영주차장 찾기')

    if 'parking_options' not in st.session_state:
        st.session_state.parking_options = None

    if 'selected_parking' not in st.session_state:
        st.session_state.selected_parking = None
    if 'map' not in st.session_state:
        st.session_state.map = None
    if 'coords' not in st.session_state:
        st.session_state.coords = []
    # 사용자 위치 입력
    user_lat =37.6545  
    user_lng =127.0492
    m1 = folium.Map(location=(user_lat,user_lng), zoom_start=16)
    if st.button('가장 가까운 공영주차장 검색'):
        parking_lots = find_nearest_parking(user_lat, user_lng)
        if parking_lots["documents"]:
            st.session_state.parking_options = [(lot['place_name'], lot['y'], lot['x']) for lot in parking_lots["documents"][:20]]  # 상위 5개 주차장 정보 추출
            
             # 선택한 주차장의 좌표를 가져와 지도에 표시
            # for lot in st.session_state.parking_options:
            #     folium.Marker(
            #         location=(float(lot[1]), float(lot[2])),
            #         popup=lot[0],
            #         icon=folium.Icon(icon='cloud')
            #     ).add_to(m1)
            # st_folium(m1, width=725, height=500)
        else:
            st.write("근처에 공영주차장이 없습니다.")
            st.session_state.parking_options = None

    if st.session_state.parking_options:
        option_names = [option[0] for option in st.session_state.parking_options]
        selected_parking_name = st.selectbox('주차장을 선택하세요:', options=option_names)
        if st.button('선택한 주차장으로 경로 표시'):
            selected_parking = next((lot for lot in st.session_state.parking_options if lot[0] == selected_parking_name), None)
            if selected_parking:
                st.session_state.selected_parking = selected_parking
                st.write(f"주소: {selected_parking[0]}")
                st.write(f"위도: {selected_parking[1]}, 경도: {selected_parking[2]}")
                coords = get_route(user_lat, user_lng, float(selected_parking[1]), float(selected_parking[2]))
                if coords:
                    st.session_state.coords = coords
                    # 지도 업데이트
    if st.session_state.coords:
        coords = st.session_state.coords  # coords를 st.session_state.coords로 할당
    if st.session_state.selected_parking:  # selected_parking가 None이 아닐 때만 업데이트
        selected_parking = st.session_state.selected_parking
        show_routes_on_map(coords, user_lat, user_lng, float(selected_parking[1]), float(selected_parking[2]))
  