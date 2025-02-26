import gradio as gr
from models.fine_tuned_model import FineTunedModel

class FineTunedInterface:
    def __init__(self):
        self.fine_tuned_model = FineTunedModel()

    def agrimind_respond(self, message, chat_history):
        bot_message = self.fine_tuned_model.chat(chat_history, message)
        chat_history.append((message, bot_message))
        return "", chat_history