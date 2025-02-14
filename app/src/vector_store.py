# vector_store.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import ZhipuAIEmbeddings
from document_loader import DocumentLoader
from config import Config

class VectorStoreManager:
    def __init__(self):
        self.document_loader = DocumentLoader()
        self.documents = self.document_loader.load_documents()
        self.vector_store = None

    def create_vector_store(self):
        """将文档转换为向量存储"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
        chunked_documents = text_splitter.split_documents(self.documents)

        embedding = ZhipuAIEmbeddings(
            model=Config.ZHIPU_EMBEDDING_MODEL,
            api_key=Config.ZHIPU_API_KEY
        )

        self.vector_store = Qdrant.from_documents(
            documents=chunked_documents,
            embedding=embedding,
            location=":memory:",
            collection_name="my_documents"
        )
        return self.vector_store
