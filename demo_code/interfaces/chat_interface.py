import gradio as gr
from models.chat_model import ChatModel

class ChatInterface:
    def __init__(self):
        self.chat_model = ChatModel()

    def chat_interface(self, history, question):
        history.append([question, None])
        response = self.chat_model.invoke(question)
        history[-1][1] = response
        return history