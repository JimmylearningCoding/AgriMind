from langchain_openai import ChatOpenAI
from config import Config
class ChatModel:
    def __init__(self):
        self.llm = ChatOpenAI(
            model='deepseek-chat',
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_API_BASE,
            max_tokens=1024
        )

    def invoke(self, prompt):
        return self.llm.invoke(prompt)