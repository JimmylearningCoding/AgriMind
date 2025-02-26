import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

class FileLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load_documents(self):
        documents = []
        for file in os.listdir(self.base_dir):
            file_path = os.path.join(self.base_dir, file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        return documents