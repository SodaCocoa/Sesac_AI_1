# 필요한 라이브러리들을 가져옵니다.
from slack_sdk.rtm_v2 import RTMClient  # Slack의 실시간 메시징을 위한 라이브러리입니다.
from slack_sdk import WebClient  # Slack과 통신하기 위한 클라이언트입니다.
from weather_info_parser import WeatherInfoParser  # 날씨 정보를 가져오는 코드가 있는 라이브러리입니다.
# import os  # 이 코드에서는 사용하지 않으므로 주석 처리했습니다.

# Slack과 통신하기 위한 클라이언트를 설정합니다. '' 안에 실제 토큰 값을 넣어야 합니다.
rtm = RTMClient(token='xoxb-6555023466485-6557214157829-hqk9aapBWIdm61E3IpM58ybg')
web_client = WebClient(token='xoxb-6555023466485-6557214157829-hqk9aapBWIdm61E3IpM58ybg')
weather_info_parser = WeatherInfoParser()  # 날씨 정보를 가져오기 위한 객체를 만듭니다.

# 날씨 정보를 Slack에 보내는 함수입니다.
def send_weather_info():
    # '창동 날씨'에 대한 날씨 정보를 가져옵니다.
    weather_info = weather_info_parser.getWeatherInfo(keyword='창동 날씨')

    # 메시지를 보낼 Slack 채널의 ID입니다. 실제 채널 ID로 바꿔주세요.
    channel_id = 'C06GG5PCT6W'

    # Slack 채널에 날씨 정보를 보냅니다. 메시지 형식은 블록으로 구성됩니다.
    rtm.web_client.chat_postMessage(
        channel=channel_id,
        blocks=[
            {'type': 'divider'},  # 구분선을 추가합니다.
            {
                'type': 'section',  # 섹션 시작
                'text': {
                    'type': 'plain_text',
                    'text': f'{weather_info.area}'  # 지역 이름을 보여줍니다.
                }
            },
            {'type': 'divider'},  # 구분선을 추가합니다.
            {
                'type': 'section',  # 섹션 시작
                'text': {
                    'type': 'plain_text',
                    'text': f"""{weather_info.weather_today}
현재기온:{weather_info.temperature_now}
최고기온:{weather_info.temperature_high}
최저기온:{weather_info.temperature_low}
"""  # 날씨 정보를 보여줍니다.
                }
            },
        ],
    )

    # '창동 날씨'에 대한 스크린샷을 가져와 파일로 저장합니다.
    weather_info_parser.getScreenshot(keyword='창동 날씨')

    # 저장된 스크린샷을 Slack 채널에 업로드합니다.
    web_client.files_upload_v2(
        channel=channel_id,
        file='info.png',  # 업로드할 파일 이름입니다.
        title='날씨 정보',  # 파일의 제목입니다.
    )

# 프로그램의 메인 함수입니다. 여기서 날씨 정보를 보내는 함수를 호출합니다.
def main():
    send_weather_info()

# 이 스크립트가 직접 실행될 때만 main 함수를 실행합니다.
if __name__ == '__main__':
    main()

# import requests
 
# def post_message(token, channel, text):
#     response = requests.post("https://slack.com/api/chat.postMessage",
#         headers={"Authorization": "Bearer "+token},
#         data={"channel": channel,"text": text}
#     )
#     print(response)