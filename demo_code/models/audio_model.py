import dashscope
from dashscope import MultiModalConversation
from config import Config
class AudioModel:
    def __init__(self):
        dashscope.api_key = Config.Audio_API_KEY

    def sound_to_text(self, audio_input):
        audio_file_path = "file://" + audio_input
        messages = [
            {"role": "system", "content": [{"text": "you are a helpful assistant"}]},
            {"role": "user", "content": [{"audio": audio_file_path}, {"text": "直接音频转文字（中文），不说废话"}]}
        ]
        response = MultiModalConversation.call(model="qwen-audio-turbo-latest", messages=messages)
        return response["output"]["choices"][0]["message"]["content"][0]["text"]