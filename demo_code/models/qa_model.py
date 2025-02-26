from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from models.chat_model import ChatModel
from utils.vector_store import VectorStore

class QAModel:
    def __init__(self):
        self.llm = ChatModel().llm
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=VectorStore().as_retriever(),
            llm=self.llm
        )
        self.qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=self.retriever)

    def query(self, question):
        result = self.qa_chain({"query": question})
        return result['result']