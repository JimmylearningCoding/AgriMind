# document_loader.py
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from config import Config

class DocumentLoader:
    def __init__(self, base_dir=Config.DOC_BASE_DIR):
        self.base_dir = base_dir
        self.documents = []

    def load_documents(self):
        """加载所有文档"""
        for file in os.listdir(self.base_dir):
            file_path = os.path.join(self.base_dir, file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                continue
            self.documents.extend(loader.load())

        return self.documents
