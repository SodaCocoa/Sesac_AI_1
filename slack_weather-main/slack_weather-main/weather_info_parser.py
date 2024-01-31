from playwright.sync_api import sync_playwright
from selectolax.parser import HTMLParser
from weather_info import WeatherInfo
import httpx

class WeatherInfoParser:
    def getScreenshot(self, keyword=str):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='chrome')
            page = browser.new_page(viewport={'width':1980, 'height':2000})

            page.goto(f'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query={keyword}')

            weather_info = page.locator('div._tab_flicking')
            weather_info.screenshot(path='info.png')

            browser.close()

    def getWeatherInfo(self, keyword=str):
        resp = httpx.get(f'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query={keyword}')

        html = HTMLParser(resp.text)
        area = html.css_first('h2.title').text()
        temperature_now = html.css_first('div._today div.temperature_text strong').text(deep=False)
        temperature_low = html.css('li.today span.lowest')[0].text(deep=False).replace('°', '')
        temperature_high = html.css('li.today span.highest')[0].text(deep=False).replace('°', '')
        weather_today = html.css_first('div._today div.weather_main span').text()

        return WeatherInfo(
            area=area,
            temperature_now=temperature_now,
            temperature_low=temperature_low,
            temperature_high=temperature_high,
            weather_today=weather_today,
        )