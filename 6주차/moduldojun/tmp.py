import streamlit as st
# Streamlit 앱의 제목 설정
def app():
    st.title('공영주차장 장애인 요금 할인 시스템')

    # 사용자 입력을 위한 사이드바 폼 생성
    with st.form(key='parking_form'):
        # 사용자로부터 주차 시간 입력 받기
        hours_parked = st.number_input('주차 시간(시간 단위)', min_value=0.0, value=1.0, step=0.5)
        # 기본 주차 요금 설정
        base_rate = st.number_input('기본 주차 요금(시간당)', min_value=0, value=2000,step=500)  # 기본값을 시간당 2000원으로 설정
        # 폼 제출 버튼
        submit_button = st.form_submit_button(label='요금 계산하기')

    # 요금 계산 및 결과 표시
    if submit_button:
        # 장애인 할인율 적용 (50% 할인)
        discount_rate = 0.5
        # 최종 요금 계산
        final_fee = (hours_parked * base_rate) * (1 - discount_rate)
        # 결과 출력
        st.write(f"총 주차 시간: {hours_parked}시간")
        st.write(f"적용된 할인율: {discount_rate * 100}%")
        st.write(f"최종 주차 요금: {final_fee}원")
        st.button('결제하기')
    
