from openai import OpenAI

class FineTunedModel:
    def __init__(self):
        self.client = OpenAI(api_key='0', base_url='http://0.0.0.0:8000/v1')

    def chat(self, history, question):
        messages = []
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": question})
        result = self.client.chat.completions.create(messages=messages, model="/group_share/models/Qwen2.5-7B-Instruct")
        return result.choices[0].message.content