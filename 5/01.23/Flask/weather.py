from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route('/weather.html', methods=['GET'])
def weather():
    # 기상청 API 호출
    api_url ="https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList?serviceKey=%3D%3D&pageNo=1&numOfRows=10&dataType=XML&dataCd=ASOS&dateCd=DAY&startDt=20100101&endDt=20100601&stnIds=108"
    response = requests.get(api_url)

    # XML 형식의 데이터를 파싱합니다.
    soup = BeautifulSoup(response.content, 'lxml')

    # 데이터를 파싱하여 필요한 정보를 추출합니다.
    weather_data = []
    for item in soup.find_all('item'):
        data = {
            'stnId': item.find('stnid').text if item.find('stnid') else '',
            'stnNm': item.find('stnnm').text if item.find('stnnm') else '',
            'tm': item.find('tm').text if item.find('tm') else '',
            'avgTa': item.find('avgta').text if item.find('avgta') else '',
            'minTa': item.find('minta').text if item.find('minta') else '',
            'maxTa': item.find('maxta').text if item.find('maxta') else '',
        }
        weather_data.append(data)

    # 파싱한 데이터를 HTML 템플릿에 전달합니다.
    return render_template('weather.html', weather_data=weather_data)

@app.route('/')
def index():
    return render_template('weather.html')  # 또는 원하는 페이지로 리디렉션

if __name__ == '__main__':
    app.run(debug=True)
