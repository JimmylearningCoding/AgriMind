import base64
from openai import OpenAI
from config import Config
class VisionModel:
    def __init__(self):
        self.client = OpenAI(
            api_key= Config.Vision_API_kEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def query(self, question, image_path):
        base64_image = self.encode_image(image_path)
        response = self.client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful and harmless assistant."}]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}},
                    {"type": "text", "text": question}
                ]}
            ],
            stream=False
        )
        return response.choices[0].message.content

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")