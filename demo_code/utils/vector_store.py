from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import ZhipuAIEmbeddings
from config import Config

class VectorStore:
    def __init__(self, chunked_documents):
        """
        初始化 VectorStore。
        
        :param chunked_documents: 分块后的文档列表
        """
        self.embedding = ZhipuAIEmbeddings(
            model='embedding-3',
            api_key=Config.Embedding_KEY
        )
        self.vectorstore = Qdrant.from_documents(
            documents=chunked_documents,  # 使用传入的分块文档
            embedding=self.embedding,
            location=":memory:",  # 使用内存存储
            collection_name="my_documents"  # 指定集合名称
        )

    def as_retriever(self):
        """
        返回向量存储的检索器。
        """
        return self.vectorstore.as_retriever()