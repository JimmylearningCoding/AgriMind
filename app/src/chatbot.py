import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from config import Config
from openai import OpenAI
from document_loader import DocumentLoader

# 带rag以及memory的多轮chatbot实现
class ChatbotWithRetrieval:
    def __init__(self,dir):
        # 加载document
        base_dir = dir
        documents = []
        for file in os.listdir(base_dir):
            file_path = os.path.join(base_dir,file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())

        # 进行文本分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20) ## 设置指定分类器
        all_splits = text_splitter.split_documents(documents)

        # 向量数据库
        embedding = ZhipuAIEmbeddings(
            model=Config.ZHIPU_EMBEDDING_MODEL,
            api_key=Config.ZHIPU_API_KEY
        )

        self.vectorstore = Qdrant.from_documents(
            documents = all_splits,
            embedding = embedding,
            location = ":memory:", # in-memory 做存储
            collection_name =  "m", # 指定collection_name
        )

        self.llm = ChatOpenAI(
            model=Config.DEEPSEEK_MODEL,
            openai_api_key=Config.DEEPSEEK_API_KEY,
            openai_api_base=Config.DEEPSEEK_API_BASE,
            max_tokens=1024
        )
        
        # 初始化Memory
        self.memory = ConversationSummaryMemory(
            llm = self.llm,
            memory_key = "chat_history",
            return_messages=True
        )

        # 设置Retrieval Chain
        
        self.retriever = MultiQueryRetriever.from_llm(
            llm = self.llm,
            retriever = self.vectorstore.as_retriever()
            
        )

        self.qa = RetrievalQA.from_chain_type(self.llm, retriever=self.retriever, memory = self.memory)
        
    
    # 交互对话的函数
    def chat_loop(self):
        print("Chatbot已经启动! 输入exit来退出程序")
        while True:
            user_input = input("你:")
            if user_input.lower() == 'exit':
                print("再见")
                break

            # 调用Retrieval Chain
            response = self.qa(user_input)
            print(f"chatbot:{response['answer']}")

if __name__ == "__main__":
    # 启动chatbot
    folder = "/group_share/LangChain_study/data"
    bot = ChatbotWithRetrieval(folder)
    bot.chat_loop()