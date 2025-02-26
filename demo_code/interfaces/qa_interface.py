from langchain.chains import RetrievalQA
from models.chat_model import ChatModel

class QAInterface:
    def __init__(self, retriever):
        """
        初始化 QAInterface。

        :param retriever: 向量存储的检索器
        """
        self.llm = ChatModel().llm
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever
        )

    def qa_interface(self, question):
        """
        处理用户查询并返回答案。

        :param question: 用户输入的问题
        :return: 模型生成的答案
        """
        result = self.qa_chain({"query": question})
        return result['result']