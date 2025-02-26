import requests
from config import Config
class WeatherModel:
    def __init__(self):
        self.QWEATHER_KEY = Config.QWEATHER_KEY

    def get_city_id(self, city_name):
        url_api_geo = f"https://geoapi.qweather.com/v2/city/lookup?location={city_name}&key={self.QWEATHER_KEY}"
        city = requests.get(url_api_geo).json()['location'][0]
        return city['id']

    def get_weather_by_coordinates(self, city_name):
        city_id = self.get_city_id(city_name)
        weather_api = f"https://devapi.qweather.com/v7/weather/now?location={city_id}&key={self.QWEATHER_KEY}"
        response_weather = requests.get(weather_api)
        weather_data = response_weather.json()
        if weather_data['code'] == '200':
            now_weather = weather_data['now']
            return f"温度：{now_weather['temp']}度\n天气：{now_weather['text']}\n湿度：{now_weather['humidity']}%\n"
        else:
            return "无法获取天气数据，请联系平台管理人员。"