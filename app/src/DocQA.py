from code import interact
import os
import base64
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Loading:加载ducument
base_dir = '/group_share/LangChain_study/data'
documents = []

for file in os.listdir(base_dir):
    file_path = os.path.join(base_dir,file)
    if file.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# Splitting:将document切分成块以便后续进行嵌入和向量存储
text_spliter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=10)
chunked_documents = text_spliter.split_documents(documents)

# Storing:将分割嵌入并存储在矢量数据库中
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.embeddings import ZhipuAIEmbeddings


embedding = ZhipuAIEmbeddings(
    model = 'embedding-3',
    api_key = '.' 
)
vectorstore = Qdrant.from_documents(
    documents=chunked_documents, # 以分块的文档
    embedding=embedding, # 用OpenAI的Embedding Model做嵌入
    location=":memory:",  # in-memory 存储
    collection_name="my_documents",) # 指定collection_name

# Retrieval 准备模型和Retrieval链
import logging # 导入Logging工具
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever # MultiQueryRetriever工具
from langchain.chains import RetrievalQA # RetrievalQA链

# 设置Logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

llm = ChatOpenAI(
    model = 'deepseek-chat',
    openai_api_key = 'sk-',
    openai_api_base = 'https://api.deepseek.com',
    max_tokens = 1024
)

retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(),llm=llm)

qa_chain = RetrievalQA.from_chain_type(llm,retriever=retriever_from_llm)
 
# Output过程
# Gradio: 定义用户交互界面
import gradio as gr
from gradio_client import Client, handle_file
# Gradio: Define the user interface

conversation_history = []

def chat_interface(history,question):
    
    # 记录用户提问
    history.append(("用户",question))

    # 构建上下文
    conversation = history[-3:]
    conversation_context = "\n".join([f"{role}: {message}" for role, message in conversation])

    response = llm.invoke(conversation_context + "\n用户: " + question)
    answer = response.content
    
    # 记录模型回答
    history.append(("AI",answer))
    return history,"" # 返回更新后的对话历史，同时清空输入框

def qa_interface(question):
    """
    Handles user query and retrieves the answer using the QA chain.
    """
    result = qa_chain({"query": question})
    return result['result']


def encode_image(image_path):
    with open(image_path,"rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def qvq_interface(question, image_path):
    if not image_path:
        return '请上传图片'
    base64_image = encode_image(image_path)
    client = OpenAI(
        api_key = "sk-",
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    response = client.chat.completions.create(
    model="qwen-vl-max",
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}
                    # "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/QVQ/demo.png"}
                },
                # {"type": "text", "text": question},
                {"type": "text", "text": question},
            ],
        }
    ],
    stream=False
    )
    print(response) 
    result = response.choices[0].message.content
    return result 


# Define the Gradio interface with custom CSS
with gr.Blocks(css="""
    .header { text-align: center; font-size: 2.5rem; color: #333; margin-bottom: 20px; }
    .button { background-color: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; font-size: 1.2rem; transition: background-color 0.3s ease; }
    .button:hover { background-color: #45a049; }
    .output { background-color: #fff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); font-size: 1.1rem; margin-top: 20px; color: #333; }
    .footer { text-align: center; color: #777; margin-top: 30px; font-size: 0.9rem; }
    }
""") as demo:
    gr.Markdown("<div class = 'header'> 🌱 AgriMind - 智能农业平台")
    
    with gr.Tabs():
        with gr.TabItem("🤖 直接和语言大模型进行聊天"):
            # gr.Markdown("### 🤖 直接和大模型进行聊天")
            chat_input = gr.Textbox(label="输入你的问题", placeholder="请输入你的问题...", lines=3)
            chat_button = gr.Button("生成答案", elem_classes="button")
            # chat_output = gr.Textbox(label="模型回答", lines=5, interactive=False)
            chat_output = gr.Chatbot(label="聊天记录")
            chat_button.click(chat_interface,inputs=[chat_output,chat_input],outputs=[chat_output,chat_input])
        
        with gr.TabItem("😀 文档回答助手"):
            # gr.Markdown("### 😀 文档回答助手")
            doc_input = gr.Textbox(label="输入你的问题", placeholder="请输入你的问题，比如：这份文档的重点是什么", lines=3)
            doc_button = gr.Button("生成答案",elem_classes="button")
            doc_output = gr.Textbox(label="模型回答", lines=5, interactive=False)
            doc_button.click(qa_interface,inputs=doc_input,outputs=doc_output)

        with gr.TabItem("🛠️ 直接和视觉大模型进行聊天"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="请上传图片", type="filepath",height=300)
                    chat_qvq_input = gr.Textbox(label="输入你的问题", placeholder="请输入你的问题", lines=3)
                    chat_qvq_button = gr.Button("生成回答", elem_classes="button")
                with gr.Column():
                    chat_qvq_output = gr.Textbox(label="模型回答", lines=8, interactive=False)
            chat_qvq_button.click(qvq_interface,inputs=[chat_qvq_input,image_input],outputs=chat_qvq_output)
            

    gr.Markdown("<div class='footer'>由 Deepseek Langchain Gradio 提供支持</div>")

# Start the Gradio app
demo.launch(server_name="0.0.0.0", server_port=5000, share=True)