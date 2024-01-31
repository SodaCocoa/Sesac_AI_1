from weather_info_parser import WeatherInfoParser
from slack_sdk.rtm_v2 import RTMClient
from slack_sdk import WebClient
from dotenv import load_dotenv
import os

# slack app token 읽어오기
load_dotenv()

# 해당 파일 내부에서만 global 하게 사용할 수 있는 변수 선언
weather_info_parser = WeatherInfoParser()
rtm = RTMClient(token=os.getenv('slack_token'))
web_client = WebClient(token=os.getenv('slack_token'))


# 키워드 입력 시, 댓글 달아줌
@rtm.on('message')
def handle(client: RTMClient, event: dict):
    keyword: str = event['text']

    if keyword.endswith('날씨'):
        weather_info = weather_info_parser.getWeatherInfo(keyword)

        channel_id = event['#날씨']

        client.web_client.chat_postMessage(
            channel = channel_id,
            blocks = [
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {
                        "type": "plain_text",
                        "text": f"{weather_info.location}",
                    }
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "text": {
                        "type": "plain_text",
                        "text": f"""{weather_info.weather_today}
현재 기온:{weather_info.temperature_now_today}
최고 기온:{weather_info.temperature_high_today}
최저 기온:{weather_info.temperature_low_today}"""
                    }
                }
	        ]
        )
    
        # 스크린샷 저장
        weather_info_parser.getScreenshot(keyword)

        # 스크린샷 업로드
        web_client.files_upload_v2(
            channel=channel_id,
            file='info.png',
            title='날씨 정보',
        )

rtm.start()
