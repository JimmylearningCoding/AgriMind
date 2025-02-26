import gradio as gr
from interfaces.chat_interface import ChatInterface
from interfaces.qa_interface import QAInterface
from interfaces.vision_interface import VisionInterface
from interfaces.weather_interface import WeatherInterface
from interfaces.audio_interface import AudioInterface
from interfaces.agent_interface import AgentInterface
from interfaces.news_interface import NewsInterface
from interfaces.fine_tuned_interface import FineTunedInterface
from utils.file_loader import FileLoader
from utils.text_splitter import TextSplitter
from utils.vector_store import VectorStore
from config import Config
import os

class AgriMindApp:
    def __init__(self):
        self.init_document_store()
        self.chat_interface = ChatInterface()
        self.qa_interface = QAInterface(retriever=self.retriever)
        self.vision_interface = VisionInterface()
        self.weather_interface = WeatherInterface()
        self.audio_interface = AudioInterface()
        self.agent_interface = AgentInterface()
        self.news_interface = NewsInterface()
        self.fine_tuned_interface = FineTunedInterface()

    def init_document_store(self):
        """
        初始化文档加载、分块和向量存储。
        """
        # 加载文档
        file_loader = FileLoader(base_dir='/group_share/LangChain_study/data')
        documents = file_loader.load_documents()

        # 切分文档
        text_splitter = TextSplitter(chunk_size=200, chunk_overlap=10)
        self.chunked_documents = text_splitter.split_documents(documents)

        # 创建向量存储
        self.vector_store = VectorStore(chunked_documents=self.chunked_documents)
        self.retriever = self.vector_store.as_retriever()

    def launch(self):
        
        with gr.Blocks(css="""
            .header {
                display: flex; 
                justify-content: space-between;  /* Distribute space between items */
                align-items: center;  /* Center items vertically */
                font-size: 3rem;  /* Adjust font size for center title */
                padding: 20px;  /* Add some padding */           
            }
            .header-left, .header-right {
                font-size: 1.2rem;
                color: #333;
            }
            .header-center {
                text-align: center;
                font-size: 5rem;
                color: #333;
            }
            .button { background-color: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 5px; font-size: 1.2rem; transition: background-color 0.3s ease; }
            .button:hover { background-color: #45a049; }
            .output { background-color: #fff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); font-size: 1.1rem; margin-top: 20px; color: #333; }
            .footer { text-align: center; color: #777; margin-top: 30px; font-size: 0.9rem; }
            .weather-section {
                display: flex;
                flex-wrap: wrap;
                gap: 30px;
                justify-content: space-between;
            }
        
            .weather-column {
                flex: 1;
                min-width: 250px;
            }
        
            .weather-button {
                background-color: #4CAF50;
                color: white;
                padding: 12px 24px;
                font-size: 1rem;
                border-radius: 8px;
                width: 100%;
                transition: transform 0.2s ease, background-color 0.3s ease;
            }
        
            .weather-button:hover {
                background-color: #45a049;
                transform: scale(1.05);
            }
        
            .weather-output {
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                font-size: 1rem;
                margin-top: 15px;
            }
            .chatbot { display: flex; flex-direction: column; }
            .chatbot .message { display: flex; margin-bottom: 10px; }
            .chatbot .message.user { justify-content: flex-start; text-align: left;}
            .chatbot .message.ai { justify-content: flex-end; text-align: right}
            .chatbot .message .avatar { width: 40px; height: 40px; border-radius: 50%; margin-right: 10px; }
            .chatbot .message .content { max-width: 70%; padding: 10px; border-radius: 10px; }
            .chatbot .message.user .content { background-color: #e1f5fe; }
            .chatbot .message.ai .content { background-color: #f5f5f5; }
                       
            .container {
                background: rgba(255, 255, 255, 0.85);
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            }
            
            .dropdown-wrapper {
                background: #fff;
                border-radius: 10px;
                padding: 12px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
        
            /* 动画效果 */
            .fade-in {
                animation: fadeIn 1s ease-in-out;
            }
        
            @keyframes fadeIn {
                0% { opacity: 0; }
                100% { opacity: 1; }
            }
        
            }
        """) as demo:
            gr.Markdown("""
            <div class = 'header'>
                <div class = 'header-left'> 🏫 Made by SCAU </div>
                <div class = 'header-center'> 👩‍🌾 AgriMind </div>
                <div class = 'header-right'> 🧑‍🎓 JimmyHo </div>
            </div>         
            """)
            with gr.Tabs():
                with gr.TabItem("🤖 直接和语言大模型进行聊天"):
                    gr.Markdown("### 简介\n在这里，你可以直接与语言大模型进行对话。输入你的问题，模型会生成回答。")

                    chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False,avatar_images=(Config.USER_AVATAR, Config.AI_AVATAR))
                    chat_input = gr.Textbox(label="输入你的问题", placeholder="请输入你的问题...", lines=3)
                    chat_button = gr.Button("生成答案")
                    chat_button.click(self.chat_interface.chat_interface, inputs=[chatbot, chat_input], outputs=[chatbot])

                with gr.TabItem("😀 文档回答助手"):
                    doc_input = gr.Textbox(label="输入你的问题", placeholder="请输入你的问题，比如：这份文档的重点是什么", lines=3)
                    doc_button = gr.Button("生成答案")
                    doc_output = gr.Textbox(label="模型回答", lines=5, interactive=False)
                    doc_button.click(self.qa_interface.qa_interface, inputs=doc_input, outputs=doc_output)

                with gr.TabItem("🛠️ 直接和视觉大模型进行聊天"):
                    image_input = gr.Image(label="请上传图片", type="filepath")
                    chat_qvq_input = gr.Textbox(label="输入你的问题", placeholder="请输入你的问题", lines=3)
                    chat_qvq_button = gr.Button("生成回答")
                    chat_qvq_output = gr.Textbox(label="模型回答", lines=8, interactive=False)
                    chat_qvq_button.click(self.vision_interface.qvq_interface, inputs=[chat_qvq_input, image_input], outputs=chat_qvq_output)

                with gr.TabItem("🌏 天气状况"):
                    city_dropdown = gr.Dropdown(label="选择城市", choices=["上海", "北京", "广州", "深圳", "成都", "其他"])
                    other_city_input = gr.Textbox(label="如果没有找到你的城市，请手动输入", placeholder="请输入城市名称", visible=False)
                    weather_button = gr.Button("获取天气信息")
                    weather_output = gr.Textbox(label="天气信息", lines=5, interactive=False)
                    weather_button.click(self.weather_interface.get_weather, inputs=[city_dropdown, other_city_input], outputs=weather_output)

                with gr.TabItem("🎙️ 语音小助手"):
                    audio_input = gr.Audio(label="请上传音频文件", type="filepath")
                    result_output = gr.Textbox(label="音频识别结果", lines=2, interactive=False)
                    process_button = gr.Button("开始语音识别")
                    llm_output = gr.Markdown(label="模型回答")
                    process_button.click(self.audio_interface.sound_to_text_response, inputs=audio_input, outputs=[result_output, llm_output])

                with gr.TabItem("😍 天气与农产品价格Agent"):
                    city_dropdown = gr.Dropdown(label="选择城市", choices=["上海", "北京", "广州", "深圳", "成都", "其他"])
                    prodcut_input = gr.Textbox(label="输入你想查询的农产品", placeholder="输入你想查询的农产品", lines=1)
                    other_city_input = gr.Textbox(label="如果没有找到你的城市，请手动输入", placeholder="请输入城市名称", visible=False)
                    agent_work_button = gr.Button("获取Agent返回的信息")
                    agent_work_output = gr.Markdown("agent回答")
                    agent_work_button.click(self.agent_interface.get_agent_info, inputs=[city_dropdown, other_city_input, prodcut_input], outputs=agent_work_output)

                with gr.TabItem("📰 相关热点新闻"):
                    news_agent_work_output = gr.Markdown("新闻Agent回答")
                    news_agent_work_button = gr.Button("获取新闻Agent返回的信息")
                    news_agent_work_button.click(self.news_interface.get_new_agent_info, outputs=news_agent_work_output)

                with gr.TabItem("🧰 使用AgriMind_qwen2.5_7b微调大模型"):
                    agrimind_chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False)
                    msg = gr.Textbox(label="输入你的问题")
                    clear = gr.Button("清除聊天内容")
                    msg.submit(self.fine_tuned_interface.agrimind_respond, [msg, agrimind_chatbot], [msg, agrimind_chatbot])
                    clear.click(lambda: [], None, agrimind_chatbot, queue=False)

        demo.queue()
        demo.launch(server_name="0.0.0.0", server_port=5000, share=True)