# inference.py
import base64
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from openai import OpenAI
from vector_store import VectorStoreManager
from config import Config

class InferenceEngine:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.DEEPSEEK_MODEL,
            openai_api_key=Config.DEEPSEEK_API_KEY,
            openai_api_base=Config.DEEPSEEK_API_BASE,
            max_tokens=1024
        )
        self.vector_store_manager = VectorStoreManager()
        self.vector_store = self.vector_store_manager.create_vector_store()
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(),
            llm=self.llm
        )
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=self.retriever)

    def chat(self, question):
        """普通聊天"""
        response = self.llm.invoke(question)
        return response.content

    def qa(self, question):
        """从文档中检索答案"""
        result = self.qa_chain({"query": question})
        return result['result']

    def encode_image(self, image_path):
        """将图像编码为Base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def image_question_answer(self, question, image_path):
        """处理图像问答"""
        if not image_path:
            return "请上传图片"

        base64_image = self.encode_image(image_path)
        client = OpenAI(
            api_key=Config.QWEN_VL_API_KEY,
            base_url=Config.QWEN_VL_API_BASE
        )

        response = client.chat.completions.create(
            model=Config.QWEN_VL_MODEL,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}},
                    {"type": "text", "text": question}
                ]}
            ],
            stream=False
        )

        return response.choices[0].message.content
