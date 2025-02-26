import gradio as gr
from models.news_model import NewsModel

class NewsInterface:
    def __init__(self):
        self.news_model = NewsModel()

    def get_new_agent_info(self):
        return self.news_model.run("近期有哪些农业新闻？")