import gradio as gr
from models.agent_model import AgentModel

class AgentInterface:
    def __init__(self):
        self.agent_model = AgentModel()

    def get_agent_info(self, city_dropdown, other_city, product):
        city = other_city if city_dropdown == '其他' else city_dropdown
        return self.agent_model.run(f"{city}的天气如何？")