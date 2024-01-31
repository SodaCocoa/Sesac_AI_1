# python-weather-api.py

import os, json
from datetime import datetime, timedelta
from urllib.parse import urlencode, unquote, quote_plus
import requests

path = os.path.dirname(os.path.abspath(__file__))
# secrets.json
{
    "token": " "
}
# 초단기예보 정보 요청
def get_ultra_srt_fcst():
    token = get_token()
    if not token:
        print("no token")
        return False
    callback_url = (
        "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
    )
    # base_date, base_time 계산
    time_now = datetime.now()
    if time_now.minute < 45:
        time_target = time_now.replace(minute=30) - timedelta(hours=1)
    else:
        time_target = time_now.replace(minute=30)
    base_date = time_target.strftime("%Y%m%d")
    base_time = time_target.strftime("%H%M")
    params = "?" + urlencode(
        {
            quote_plus("serviceKey"): token,  # 인증키
            quote_plus("numOfRows"): "60",  # 한 페이지 결과 수 // default : 10
            quote_plus("pageNo"): "1",  # 페이지 번호 // default : 1
            quote_plus("dataType"): "JSON",  # 응답자료형식 : XML, JSON
            quote_plus("base_date"): base_date,  # 발표일자 // yyyymmdd
            quote_plus("base_time"): base_time,  # 발표시각 // HHMM, 매 시각 45분 이후 호출
            quote_plus("nx"): "60",  # 예보지점 X 좌표
            quote_plus("ny"): "127",  # 예보지점 Y 좌표
        }
    )
    res = requests.get(callback_url + unquote(params))
    items = res.json().get("response").get("body").get("items").get("item")
    print(items)