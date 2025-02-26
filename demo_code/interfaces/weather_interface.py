import gradio as gr
from models.weather_model import WeatherModel

class WeatherInterface:
    def __init__(self):
        self.weather_model = WeatherModel()

    def get_weather(self, city_dropdown, other_city):
        city = other_city if city_dropdown == "其他" else city_dropdown
        return self.weather_model.get_weather_by_coordinates(city)