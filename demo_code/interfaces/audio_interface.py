import gradio as gr
from models.audio_model import AudioModel
from models.chat_model import ChatModel

class AudioInterface:
    def __init__(self):
        self.audio_model = AudioModel()
        self.chat_model = ChatModel()

    def sound_to_text_response(self, audio_input):
        text = self.audio_model.sound_to_text(audio_input)
        answer = self.chat_model.invoke(text)
        return text, answer